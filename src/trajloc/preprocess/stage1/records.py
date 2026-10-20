"""The per-clip records every Stage-1 builder writes, built in one place.

A clip's records:

- ``videos/<stem>.mp4`` and ``first_frames/<stem>.png``
  (:func:`write_clip_media`);
- ``trajectories/<stem>.json`` (:func:`trajectory_entry`)::

      {"objects": {"0": {"label": "dog", "source_object_id": 3,
                         "positions": [[0.41, 0.52], null, ...],      # 49 frames, [x, y]
                         "boxes": [[0.30, 0.40, 0.20, 0.25], null, ...],  # [x, y, w, h]
                         "depths": [0.0, null, ...]}}}

  Coordinates are normalized to [0, 1]; ``null`` where the object is not
  visible; depth is 0 wherever it is. ``source_object_id`` is the object's
  index in the source annotation (SVG2's; for the benchmarks, the
  foreground objects, then the background ones);
- ``prompts/<stem>.json`` (:func:`prompt_record`)::

      {"prompt": "A person in red <traj_0> walks past a metallic pole <traj_1>. ...",
       "objects": [{"object_index": 0, "source_object_id": 3, "noun": "person",
                    "modifier": "in red", "noun_word_index": 1,
                    "trajectory_token_word_index": 4}, ...],
       "words_located": true, "word_location_errors": []}

  The word indices point into ``prompt.split()``. A dataset root's prompts
  append the scene caption and keep both halves (``object_prompt``,
  ``scene_caption``; :mod:`.caption_join`).

Object ``i`` of the trajectories and ``objects[i]`` of the prompt describe
the same object. :func:`identity_error` is the check, run before anything is
written and again by Stage 2 before a clip is cached.

These records are written with :func:`write_json` (``json.dump`` defaults:
ASCII-escaped; atomic). The caption joins write theirs with
:func:`trajloc.preprocess.io.atomic_write_json` (UTF-8, non-ASCII kept;
atomic). Each builder keeps its own format.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from trajloc.datasets.layout import CLIP_FPS, CLIP_HEIGHT, CLIP_NUM_FRAMES, CLIP_WIDTH

COORD_DECIMALS = 6  # normalized coordinates are rounded to this many decimals

PROMPT_OBJECT_KEYS = (
    "object_index",
    "source_object_id",
    "noun",
    "modifier",
    "noun_word_index",
    "trajectory_token_word_index",
)


def normalized_positions(
    rx: np.ndarray, ry: np.ndarray, valid: np.ndarray
) -> list[list[float] | None]:
    """Per frame, ``[x, y]`` in [0, 1] (clip pixels / frame size), or
    ``None`` where the trajectory is not valid or not finite."""
    positions: list[list[float] | None] = []
    for t in range(CLIP_NUM_FRAMES):
        if not valid[t] or not (np.isfinite(rx[t]) and np.isfinite(ry[t])):
            positions.append(None)
        else:
            positions.append(
                [
                    round(float(rx[t] / CLIP_WIDTH), COORD_DECIMALS),
                    round(float(ry[t] / CLIP_HEIGHT), COORD_DECIMALS),
                ]
            )
    return positions


def normalized_boxes(masks_t: list[np.ndarray | None] | np.ndarray) -> list[list[float] | None]:
    """Per frame, the mask's bounding box ``[x, y, w, h]`` in [0, 1], or
    ``None`` where the mask is missing or empty.

    ``masks_t`` holds ``CLIP_NUM_FRAMES`` boolean or 0/1 masks (or ``None``).
    ``w`` and ``h`` are ``max - min`` of the mask's pixel coordinates.
    """
    boxes: list[list[float] | None] = []
    for t in range(CLIP_NUM_FRAMES):
        m = masks_t[t]
        if m is None or not m.any():
            boxes.append(None)
            continue
        ys, xs = np.where(m)
        bx, by = float(xs.min()), float(ys.min())
        bw = float(xs.max() - xs.min())
        bh = float(ys.max() - ys.min())
        boxes.append(
            [
                round(bx / CLIP_WIDTH, COORD_DECIMALS),
                round(by / CLIP_HEIGHT, COORD_DECIMALS),
                round(bw / CLIP_WIDTH, COORD_DECIMALS),
                round(bh / CLIP_HEIGHT, COORD_DECIMALS),
            ]
        )
    return boxes


def zero_depths(positions: list) -> list:
    """Depth 0 wherever the position is set (the released data has no depth)."""
    return [None if p is None else 0.0 for p in positions]


def trajectory_entry(label: str, source_object_id: int, positions: list, boxes: list) -> dict:
    """One object of ``trajectories/<stem>.json``."""
    return {
        "label": label,
        "source_object_id": int(source_object_id),
        "positions": positions,
        "boxes": boxes,
        "depths": zero_depths(positions),
    }


def identity_error(prompt_objects: list[dict], trajectory_objects: dict[str, dict]) -> str | None:
    """Why a clip's prompt objects and its trajectories don't describe the
    same objects in the same order, or ``None`` if they do.

    The one definition of that agreement, for the Stage-1 writers (before
    anything is written) and for Stage 2 (before a cache is built, through
    :func:`check_identity`):

    - there are as many prompt objects as trajectories;
    - the trajectory keys are ``"0"`` .. ``"n-1"``;
    - ``prompt_objects[i]["object_index"] == i``;
    - ``prompt_objects[i]`` and trajectory ``i`` have the same ``source_object_id``.
    """
    num_kept = len(trajectory_objects)
    if len(prompt_objects) != num_kept:
        return f"prompt objects={len(prompt_objects)} != trajectories={num_kept}"
    keys = sorted(trajectory_objects, key=int)
    if [int(k) for k in keys] != list(range(num_kept)):
        return f"trajectories key sequence {keys} is not [0..n)"
    for i, (m, key) in enumerate(zip(prompt_objects, keys)):
        if m["object_index"] != i:
            return f"prompt[{i}].object_index={m['object_index']} != {i}"
        prompt_source_id = m.get("source_object_id")
        trajectory_source_id = trajectory_objects[key].get("source_object_id")
        if prompt_source_id is None or trajectory_source_id is None:
            return (
                f"missing source_object_id (trajectories={trajectory_source_id}, "
                f"prompt={prompt_source_id})"
            )
        if int(prompt_source_id) != int(trajectory_source_id):
            return (
                f"prompt[{i}].source_object_id={prompt_source_id} != "
                f"trajectories[{i}].source_object_id={trajectory_source_id}"
            )
    return None


def check_identity(prompt_record: dict, trajectories_record: dict, stem: str) -> None:
    """Raise ``RuntimeError`` if ``prompts/<stem>.json`` and
    ``trajectories/<stem>.json`` disagree (:func:`identity_error`).

    A disagreement would silently attach trajectories to the wrong objects'
    words (e.g. a ``prompts/`` directory reused from a Stage-1 run that kept
    a different set of objects).
    """
    error = identity_error(prompt_record["objects"], trajectories_record["objects"])
    if error is not None:
        raise RuntimeError(f"prompt and trajectories of {stem} describe different objects: {error}")


@dataclass(frozen=True)
class PromptSynthesis:
    """What a prompt builder produces for a clip (both builders:
    :func:`~.sav.prompts.build_scene_graph_prompt`,
    :func:`~.bench.prompts.build_benchmark_prompt`).

    Attributes:
        prompt: the sentence, with a ``<traj_i>`` token per object.
        objects: per object, in ``object_index`` order, the
            :data:`PROMPT_OBJECT_KEYS` (word indices into ``prompt.split()``,
            ``None`` where not found).
        ok: whether every object's words were found.
        errors: why not.
    """

    prompt: str
    objects: list[dict]
    ok: bool
    errors: list[str]


def prompt_record(synthesis: PromptSynthesis, **extra: Any) -> dict:
    """``prompts/<stem>.json`` from a builder's result; ``extra`` keys go last."""
    return {
        "prompt": synthesis.prompt,
        "objects": [{k: m[k] for k in PROMPT_OBJECT_KEYS} for m in synthesis.objects],
        "words_located": synthesis.ok,
        "word_location_errors": synthesis.errors,
        **extra,
    }


def write_json(path: Path, payload: Any) -> None:
    """``json.dump(payload)`` to ``path`` (the format of the Stage-1 records),
    atomically: resuming steps take an existing record as done."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.chmod(tmp, 0o644)  # mkstemp creates it 0600: a normal file's mode
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_clip_media(
    frames_bgr: list[np.ndarray], video_path: Path, first_frame_path: Path
) -> None:
    """The clip's frames (BGR, clip size) as an mp4v video at
    ``CLIP_FPS``, and its first frame as a PNG."""
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter.fourcc(*"mp4v"), CLIP_FPS, (CLIP_WIDTH, CLIP_HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for {video_path} (is mp4v available?)")
    for fr in frames_bgr:
        writer.write(fr)
    writer.release()
    cv2.imwrite(str(first_frame_path), frames_bgr[0])
