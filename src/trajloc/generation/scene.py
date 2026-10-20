"""A user's scene: a prompt and one trajectory per object, parsed and checked.

A scene file (JSON) states what to generate from an image::

    {"prompt": "A corgi <traj_0> trots across the lawn toward a red ball <traj_1>.",
     "objects": [
        {"noun": "corgi", "keyframes": [[0, 0.30, 0.62], [48, 0.62, 0.58]]},
        {"noun": "ball",  "positions": [[0.71, 0.66], [0.71, 0.66], ..., null]}]}

- ``prompt``: the text, with one ``<traj_i>`` token per object, written
  after the object's words (``objects[i]`` is ``<traj_i>``);
- ``noun``: the object's head noun (one or more words), as it appears
  before its ``<traj_i>``; its appearance token goes there;
- the path, as exactly one of ``positions`` (49 entries: ``[x, y]``, or
  ``null`` where the object is not visible) or ``keyframes`` (``[frame, x,
  y]`` rows, frames increasing in [0, 48], linearly interpolated between;
  ``null`` before the first and after the last).

Coordinates are normalized to the input image, ``(0, 0)`` its top-left and
``(1, 1)`` its bottom-right corner. The image is center-cropped to 3:2 and
resized to 480x720 (HxW) like every dataset's clips
(:mod:`trajloc.preprocess.stage1.geometry`), so every point must lie in that
crop.

:func:`load_scene` parses and checks a scene; :meth:`Scene.records` gives
the Stage-1 ``prompts/`` and ``trajectories/`` records Stage 2 reads.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trajloc.core.trajectory_encoder import NUM_TRAJECTORY_TOKENS
from trajloc.datasets.layout import CLIP_HEIGHT, CLIP_NUM_FRAMES, CLIP_WIDTH
from trajloc.preprocess.stage1.geometry import CropWindow, center_crop_window
from trajloc.preprocess.stage1.prompt_words import locate_object_words
from trajloc.preprocess.stage1.records import (
    PromptSynthesis,
    identity_error,
    normalized_positions,
    prompt_record,
    trajectory_entry,
)

_TRAJECTORY_TOKEN = re.compile(r"<traj_(\d+)>")

Point = tuple[float, float]


class SceneError(ValueError):
    """A scene file that can't be generated, and why."""


@dataclass(frozen=True)
class SceneObject:
    """One object: its head noun and, per frame, its position in the input
    image (normalized; ``None`` where not visible)."""

    noun: str
    positions: tuple[Point | None, ...]


@dataclass(frozen=True)
class Scene:
    """A checked scene (see the module docstring)."""

    prompt: str
    objects: tuple[SceneObject, ...]

    def records(self, image_width: int, image_height: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """The ``(prompts, trajectories)`` Stage-1 records of this scene over
        an ``image_width x image_height`` image
        (:mod:`trajloc.preprocess.stage1.records`).

        Positions are mapped through the image's center crop into the clip
        frame. Raises :class:`SceneError` if one lies outside the crop.
        """
        window = center_crop_window(image_width, image_height)
        words = self.prompt.split()
        prompt_objects, trajectories = [], {}
        for i, obj in enumerate(self.objects):
            trajectory_word, noun_word = locate_object_words(words, i, obj.noun)
            prompt_objects.append(
                {
                    "object_index": i,
                    "source_object_id": i,
                    "noun": obj.noun,
                    "modifier": "",
                    "noun_word_index": noun_word,
                    "trajectory_token_word_index": trajectory_word,
                }
            )
            positions = _clip_positions(obj, i, window, image_width, image_height)
            boxes = [None] * CLIP_NUM_FRAMES  # boxes weigh the training loss only
            trajectories[str(i)] = trajectory_entry(obj.noun, i, positions, boxes)
        synthesis = PromptSynthesis(prompt=self.prompt, objects=prompt_objects, ok=True, errors=[])
        prompts = prompt_record(synthesis)
        error = identity_error(prompts["objects"], trajectories)
        if error is not None:  # the records are built together: a bug, not a user error
            raise AssertionError(error)
        return prompts, {"objects": trajectories}


def load_scene(path: str | Path) -> Scene:
    """Parse and check the scene file at ``path``; :class:`SceneError` if it is invalid."""
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise SceneError(f"{path}: not valid JSON ({e})") from None
    return parse_scene(data, source=str(path))


def parse_scene(data: Any, *, source: str = "scene") -> Scene:
    """A :class:`Scene` from a scene file's parsed JSON."""
    if not isinstance(data, Mapping):
        raise SceneError(f"{source}: expected an object with 'prompt' and 'objects'")
    _check_keys(data, {"prompt", "objects"}, source)
    prompt, entries = data["prompt"], data["objects"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise SceneError(f"{source}: 'prompt' must be a non-empty string")
    if not isinstance(entries, list) or not entries:
        raise SceneError(f"{source}: 'objects' must be a non-empty list")
    if len(entries) > NUM_TRAJECTORY_TOKENS:
        raise SceneError(f"{source}: at most {NUM_TRAJECTORY_TOKENS} objects")
    _check_trajectory_tokens(prompt, len(entries), source)

    words = prompt.split()
    objects = []
    for i, entry in enumerate(entries):
        where = f"{source}: objects[{i}]"
        if not isinstance(entry, Mapping):
            raise SceneError(f"{where}: expected an object")
        _check_keys(
            entry, {"noun", "positions", "keyframes"}, where, optional={"positions", "keyframes"}
        )
        noun = entry["noun"]
        if not isinstance(noun, str) or not noun.split():
            raise SceneError(f"{where}: 'noun' must be a non-empty string")
        if locate_object_words(words, i, noun)[1] is None:
            raise SceneError(f"{where}: the noun {noun!r} is not in the prompt before <traj_{i}>")
        if ("positions" in entry) == ("keyframes" in entry):
            raise SceneError(f"{where}: give exactly one of 'positions' and 'keyframes'")
        if "positions" in entry:
            positions = _parse_positions(entry["positions"], where)
        else:
            positions = _interpolate_keyframes(entry["keyframes"], where)
        if all(p is None for p in positions):
            raise SceneError(f"{where}: the object is visible in no frame")
        objects.append(SceneObject(noun=" ".join(noun.split()), positions=positions))
    return Scene(prompt=prompt, objects=tuple(objects))


# ----------------------------------------------------------------- internals


def _check_keys(
    values: Mapping[str, Any], accepted: Set[str], where: str, *, optional: Set[str] = frozenset()
) -> None:
    unknown = sorted(set(values) - accepted)
    missing = sorted(accepted - optional - set(values))
    if unknown or missing:
        raise SceneError(f"{where}: unknown keys {unknown}, missing keys {missing}")


def _check_trajectory_tokens(prompt: str, num_objects: int, source: str) -> None:
    """Each of ``<traj_0>`` .. ``<traj_{n-1}>`` once, as a word of its own, and no other."""
    found = [int(m) for m in _TRAJECTORY_TOKEN.findall(prompt)]
    expected = list(range(num_objects))
    if sorted(found) != expected:
        raise SceneError(
            f"{source}: the prompt must have each of <traj_0> .. <traj_{num_objects - 1}> "
            f"exactly once (one per object); it has {[f'<traj_{i}>' for i in found]}"
        )
    words = {w.rstrip(",.") for w in prompt.split()}
    glued = [f"<traj_{i}>" for i in expected if f"<traj_{i}>" not in words]
    if glued:
        raise SceneError(
            f"{source}: {glued} must be words of their own (spaces around them; only a "
            f"comma or period may follow)"
        )


def _parse_point(value: Any, where: str) -> Point:
    if (
        not isinstance(value, Sequence)
        or len(value) != 2
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
        or not all(math.isfinite(v) for v in value)
    ):
        raise SceneError(f"{where}: expected [x, y], got {value!r}")
    x, y = float(value[0]), float(value[1])
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise SceneError(
            f"{where}: [x, y] must be in [0, 1] (normalized to the image), got {value!r}"
        )
    return x, y


def _parse_positions(value: Any, where: str) -> tuple[Point | None, ...]:
    if not isinstance(value, list) or len(value) != CLIP_NUM_FRAMES:
        raise SceneError(
            f"{where}: 'positions' must have {CLIP_NUM_FRAMES} entries (one per frame)"
        )
    return tuple(
        None if p is None else _parse_point(p, f"{where}.positions[{t}]")
        for t, p in enumerate(value)
    )


def _interpolate_keyframes(value: Any, where: str) -> tuple[Point | None, ...]:
    if not isinstance(value, list) or not value:
        raise SceneError(f"{where}: 'keyframes' must be a non-empty list of [frame, x, y]")
    frames, points = [], []
    for k, row in enumerate(value):
        if (
            not isinstance(row, Sequence)
            or len(row) != 3
            or not isinstance(row[0], int)
            or isinstance(row[0], bool)
        ):
            raise SceneError(f"{where}.keyframes[{k}]: expected [frame, x, y], got {row!r}")
        frames.append(row[0])
        points.append(_parse_point(row[1:], f"{where}.keyframes[{k}]"))
    if any(not 0 <= f < CLIP_NUM_FRAMES for f in frames) or frames != sorted(set(frames)):
        raise SceneError(
            f"{where}: keyframe frames must increase, in [0, {CLIP_NUM_FRAMES - 1}]; got {frames}"
        )
    xs = np.interp(np.arange(CLIP_NUM_FRAMES), frames, [p[0] for p in points])
    ys = np.interp(np.arange(CLIP_NUM_FRAMES), frames, [p[1] for p in points])
    return tuple(
        (float(xs[t]), float(ys[t])) if frames[0] <= t <= frames[-1] else None
        for t in range(CLIP_NUM_FRAMES)
    )


def _clip_positions(
    obj: SceneObject, index: int, window: CropWindow, image_width: int, image_height: int
) -> list[list[float] | None]:
    """The object's positions in the clip frame, as the records store them."""
    valid = np.array([p is not None for p in obj.positions])
    px = np.array([p[0] * image_width if p is not None else np.nan for p in obj.positions])
    py = np.array([p[1] * image_height if p is not None else np.nan for p in obj.positions])
    cx, cy = window.to_clip_pixels(px, py, dtype=np.float64)
    outside = [
        t for t in range(CLIP_NUM_FRAMES)
        if valid[t] and not (0 <= cx[t] <= CLIP_WIDTH and 0 <= cy[t] <= CLIP_HEIGHT)
    ]  # fmt: skip
    if outside:
        raise SceneError(
            f"objects[{index}] ({obj.noun!r}): frames {outside} lie outside the image's "
            f"center 3:2 crop, which is what gets generated"
        )
    corner = [t for t in range(CLIP_NUM_FRAMES) if valid[t] and cx[t] <= 0 and cy[t] <= 0]
    if corner:  # the heatmaps read (0, 0) as "not visible"
        raise SceneError(
            f"objects[{index}] ({obj.noun!r}): frames {corner} are at the frame's top-left "
            f"corner (0, 0), which the model reads as not visible; move them inward"
        )
    return normalized_positions(cx, cy, valid)
