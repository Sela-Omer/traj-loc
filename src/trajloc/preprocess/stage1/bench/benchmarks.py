"""The benchmarks: everything the benchmark builders know about each one.

The builders (:mod:`.object_tracking`, :mod:`.object_naming` and the caption
join) are shared. A :class:`Benchmark` supplies the parts in which the
benchmarks differ:

- where its data is (:attr:`Benchmark.source`, a reader from :mod:`.sources`);
- its foreground objects for a clip. This is a template method:
  :meth:`~Benchmark.read_annotation` loads and checks the clip's annotation,
  the tracking step writes the clip's frames for SAM 2, then
  :meth:`~Benchmark.clip_masks` and :meth:`~Benchmark.trajectories` turn
  the annotation into masks and (optionally) trajectories;
- the leading fields of its ``objects/`` records.

MoveBench also ships a caption per clip (:func:`load_movebench_captions`).

To add a benchmark: a reader in :mod:`.sources` and a :class:`Benchmark`
here; ``track_objects.py``, ``name_objects.py`` and ``run_stage1.sh``
scripts like MagicBench's; its captions and ``join_captions.py``; then
Stage 2 and an evaluation set (:mod:`trajloc.datasets.eval_sets`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trajloc.datasets.layout import CLIP_HEIGHT, CLIP_NUM_FRAMES, CLIP_WIDTH
from trajloc.preprocess.stage1.bench.clip_geometry import ground_truth_trajectory
from trajloc.preprocess.stage1.bench.segmentation import Sam2Models, propagate_masks
from trajloc.preprocess.stage1.bench.sources import MagicBenchSource, MoveBenchSource, VideoSource
from trajloc.preprocess.stage1.geometry import CropWindow
from trajloc.preprocess.stage1.tracking import Trajectory


class ClipSkipped(Exception):
    """The clip's annotation can't be used. Stage 1A records
    ``{"stem", "error": <message>}`` in ``errors/<stem>.err`` and goes on."""


class Benchmark(ABC):
    """One benchmark's data and foreground annotation."""

    source: VideoSource

    @abstractmethod
    def read_annotation(self, stem: str, H: int, W: int) -> Any:
        """Load the clip's foreground annotation (``H x W`` is the video's
        native size) and check it; raise :class:`ClipSkipped` if it can't be
        used. Runs before the clip's frames are written."""

    @abstractmethod
    def clip_masks(
        self, annotation: Any, crop: CropWindow, frames_dir: Path, sam2: Sam2Models
    ) -> np.ndarray:
        """The foreground objects' ``(n_fg, 49, 480, 720)`` bool masks.
        ``frames_dir`` holds the clip's frames, for SAM 2."""

    def trajectories(
        self, annotation: Any, picks: list[int], crop: CropWindow
    ) -> list[Trajectory | None] | None:
        """The foreground objects' own trajectories (``None`` for an object
        with none, which is then dropped), or ``None`` to track them all."""
        return None

    @abstractmethod
    def objects_record_header(self, stem: str) -> dict:
        """The leading fields of ``objects/<stem>.json``."""


@dataclass(frozen=True)
class _MoveBenchAnnotation:
    frame0_masks: list[np.ndarray]  # per object, bool, native
    tracks: np.ndarray  # (T, n_fg, 2) source pixels
    visibility: np.ndarray  # (T, n_fg) bool


class MoveBench(Benchmark):
    """MoveBench (``en/``): frame-0 object masks, propagated with SAM 2, and
    ground-truth point tracks, used verbatim as the trajectories."""

    source: MoveBenchSource

    def __init__(self, root: Path) -> None:
        self.source = MoveBenchSource(Path(root))

    def read_annotation(self, stem: str, H: int, W: int) -> _MoveBenchAnnotation:
        _rgb0, _fg_union, masks, _ids = self.source.load_frame0_with_fg(stem)
        tracks, vis = self.source.load_gt_tracks(stem)
        if tracks.shape[1] != len(masks):
            raise ClipSkipped(
                f"track_mask_count_mismatch:tracks={tracks.shape[1]}!=masks={len(masks)}"
            )
        return _MoveBenchAnnotation(masks, tracks, vis)

    def clip_masks(
        self,
        annotation: _MoveBenchAnnotation,
        crop: CropWindow,
        frames_dir: Path,
        sam2: Sam2Models,
    ) -> np.ndarray:
        frame0: list[np.ndarray] = []
        for source_mask in annotation.frame0_masks:
            clip_mask = crop.warp_mask_bool(source_mask)
            frame0.append(
                clip_mask if clip_mask.any() else np.zeros((CLIP_HEIGHT, CLIP_WIDTH), bool)
            )
        return propagate_masks(
            sam2.video_predictor, frames_dir, frame0, empty_hw=(CLIP_HEIGHT, CLIP_WIDTH)
        )

    def trajectories(
        self, annotation: _MoveBenchAnnotation, picks: list[int], crop: CropWindow
    ) -> list[Trajectory | None]:
        return [
            ground_truth_trajectory(annotation.tracks, annotation.visibility, i, picks, crop)
            for i in range(len(annotation.frame0_masks))
        ]

    def objects_record_header(self, stem: str) -> dict:
        return {"stem": stem, "subset": self.source.subset}


class MagicBench(Benchmark):
    """MagicBench: per-frame instance masks; no tracks (all objects are
    tracked) and no captions (its scene captions are Qwen2.5-VL's)."""

    source: MagicBenchSource

    def __init__(self, root: Path) -> None:
        self.source = MagicBenchSource(Path(root))

    def read_annotation(self, stem: str, H: int, W: int) -> list[np.ndarray]:
        stacks, _colors, mH, mW = self.source.load_instances_per_frame(stem)
        if (mH, mW) != (H, W):
            raise ClipSkipped(f"mask_video_res_mismatch:mask={(mH, mW)}!=video={(H, W)}")
        return stacks

    def clip_masks(
        self,
        annotation: list[np.ndarray],
        crop: CropWindow,
        frames_dir: Path,
        sam2: Sam2Models,
    ) -> np.ndarray:
        masks = np.zeros((len(annotation), CLIP_NUM_FRAMES, CLIP_HEIGHT, CLIP_WIDTH), dtype=bool)
        for k, stack in enumerate(annotation):
            for t in range(CLIP_NUM_FRAMES):
                masks[k, t] = crop.warp_mask_bool(stack[t])
        return masks

    def objects_record_header(self, stem: str) -> dict:
        return {"stem": stem}


def load_movebench_captions(track_txt: Path) -> dict[str, str]:
    """MoveBench's ``<subset>_track.txt`` (``<stem>.mp4,<caption>`` lines)
    as ``stem -> caption``."""
    out: dict[str, str] = {}
    if not track_txt.is_file():
        raise FileNotFoundError(f"no MoveBench caption list at {track_txt}")
    with open(track_txt) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            i = line.find(",")
            if i < 0:
                continue
            out[line[:i].strip().replace(".mp4", "")] = line[i + 1 :].strip()
    return out
