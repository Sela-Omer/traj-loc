"""Reading the benchmarks' source data.

- **MoveBench** (``<root>`` = ``MoveBench/en``): ``video/<stem>.mp4`` (832x480,
  81 frames, 16 fps), frame-0 object masks
  ``first_frame_mask/<subset>/<stem>_mask_<i>.png``, the frame
  ``first_frame/<stem>.jpg``, and ground-truth point tracks
  ``track/<subset>/<stem>_{tracks,visibility}.npy`` (one track per object,
  column ``j`` <-> mask ``j + 1``). These tracks are the benchmark's
  conditioning signal, used verbatim as the foreground trajectories.
- **MagicBench** (``<root>``: the MagicBench release): ``videos/<id>.mp4``
  (1920x1080 or 1920x1012, 49 frames) and per-frame instance masks
  ``masks/<id>/annotated_frame_NNNNN.png``: one solid color per instance,
  stable over the clip, on black. No tracks or captions.

Frames are returned as BGR at native resolution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
from PIL import Image

from trajloc.datasets.layout import CLIP_NUM_FRAMES


class VideoSource(ABC):
    """A benchmark's clips: where each clip's video is, and reading it."""

    root: Path

    @abstractmethod
    def video_path(self, stem: str) -> Path:
        """The clip's mp4."""

    def read_video_frames(self, stem: str) -> tuple[list[np.ndarray], int, int]:
        """All frames of the clip's mp4 as BGR uint8, and ``(H, W)``."""
        p = self.video_path(stem)
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open mp4: {p}")
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames decoded from {p}")
        H_native, W_native = frames[0].shape[:2]
        return frames, H_native, W_native


@dataclass(frozen=True)
class MoveBenchSource(VideoSource):
    """MoveBench's ``en/`` release (see the module docstring), its
    multi-object subset (the evaluation set)."""

    root: Path
    subset: ClassVar[str] = "multi"

    def video_path(self, stem: str) -> Path:
        return self.root / "video" / f"{stem}.mp4"

    def first_frame_path(self, stem: str) -> Path:
        """The clip's first frame, the image its masks were drawn on."""
        return self.root / "first_frame" / f"{stem}.jpg"

    def fg_mask_paths(self, stem: str) -> list[Path]:
        """``<stem>_mask_<i>.png`` sorted by ``i`` numerically."""
        d = self.root / "first_frame_mask" / self.subset
        return sorted(d.glob(f"{stem}_mask_*.png"), key=lambda p: int(p.stem.split("_mask_")[-1]))

    def load_gt_tracks(self, stem: str) -> tuple[np.ndarray, np.ndarray]:
        """``(tracks (T, N, 2) float32 native px, visibility (T, N) bool)``."""
        tr = np.load(self.root / "track" / self.subset / f"{stem}_tracks.npy")
        vis = np.load(self.root / "track" / self.subset / f"{stem}_visibility.npy")
        if tr.ndim == 4:
            tr = tr[0]
        if vis.ndim == 3:
            vis = vis[0]
        return tr.astype(np.float32), vis.astype(bool)

    def load_frame0_with_fg(
        self, stem: str
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[int]]:
        """``(rgb0, fg_union, per-object frame-0 masks (bool, native), mask ids)``,
        objects in mask-index order (the tracks' column order)."""
        rgb = np.array(Image.open(self.first_frame_path(stem)).convert("RGB"))
        H, W = rgb.shape[:2]
        paths = self.fg_mask_paths(stem)
        if not paths:
            raise RuntimeError(f"no FG masks for stem {stem} in subset {self.subset}")
        obj_ids: list[int] = []
        masks: list[np.ndarray] = []
        for p in paths:
            idx_str = p.stem.split("_mask_")[-1]
            try:
                idx = int(idx_str)
            except ValueError:
                raise RuntimeError(f"can't parse mask idx from {p.name}") from None
            m = np.array(Image.open(p)) > 0
            if m.ndim == 3:
                m = m.any(axis=-1)
            if m.shape != (H, W):
                raise RuntimeError(f"mask {p.name} shape {m.shape} != frame shape {(H, W)}")
            obj_ids.append(idx)
            masks.append(m)
        fg_union = np.zeros((H, W), dtype=bool)
        for m in masks:
            fg_union |= m
        return rgb, fg_union, masks, obj_ids


@dataclass(frozen=True)
class MagicBenchSource(VideoSource):
    """The MagicBench release (see the module docstring)."""

    root: Path

    def video_path(self, stem: str) -> Path:
        return self.root / "videos" / f"{stem}.mp4"

    def mask_frame_paths(self, stem: str) -> list[Path]:
        """``annotated_frame_NNNNN.png`` sorted by ``NNNNN``."""
        d = self.root / "masks" / stem
        return sorted(d.glob("annotated_frame_*.png"), key=lambda p: int(p.stem.split("_")[-1]))

    def instance_colors_frame0(self, stem: str) -> list[tuple[int, int, int]]:
        """Frame 0's non-black colors, sorted: instance ``i`` is color ``i``."""
        paths = self.mask_frame_paths(stem)
        if not paths:
            raise RuntimeError(f"no mask frames for clip {stem}")
        m0 = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
        if m0 is None:
            raise RuntimeError(f"failed to read mask {paths[0]}")
        if m0.ndim != 3 or m0.shape[2] < 3:
            raise RuntimeError(f"unexpected mask shape {m0.shape} for {paths[0]}")
        flat = m0[..., :3].reshape(-1, 3)
        uniq = np.unique(flat, axis=0)
        colors = [(int(c[0]), int(c[1]), int(c[2])) for c in uniq.tolist() if tuple(c) != (0, 0, 0)]
        return sorted(colors)

    def load_instances_per_frame(
        self, stem: str
    ) -> tuple[list[np.ndarray], list[tuple[int, int, int]], int, int]:
        """Per instance, its ``(49, H, W)`` bool masks (exact color match; the
        last mask frame repeats if there are fewer), the colors, and ``H, W``."""
        paths = self.mask_frame_paths(stem)
        colors = self.instance_colors_frame0(stem)
        num_objects = len(colors)
        if num_objects == 0:
            raise RuntimeError(f"no FG instances (frame-0 all black) for clip {stem}")

        m0 = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
        H, W = m0.shape[:2]

        per_obj = [np.zeros((CLIP_NUM_FRAMES, H, W), dtype=bool) for _ in range(num_objects)]
        color_arr = np.array(colors, dtype=np.int32)  # (num_objects, 3)
        for t in range(CLIP_NUM_FRAMES):
            src_idx = min(t, len(paths) - 1)
            m = cv2.imread(str(paths[src_idx]), cv2.IMREAD_UNCHANGED)
            if m is None:
                raise RuntimeError(f"failed to read mask {paths[src_idx]}")
            m = m[..., :3].astype(np.int32)
            if m.shape[:2] != (H, W):
                raise RuntimeError(
                    f"mask frame {paths[src_idx].name} shape {m.shape[:2]} != {(H, W)}"
                )
            for k in range(num_objects):
                per_obj[k][t] = np.all(m == color_arr[k], axis=-1)
        return per_obj, colors, H, W
