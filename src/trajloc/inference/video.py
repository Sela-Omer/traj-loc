"""Video file output."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from trajloc.backbones.base import VideoFrames

GENERATED_FPS = 8  # the frame rate generated videos are written at


def write_video(path: Path, frames: VideoFrames, fps: int = GENERATED_FPS) -> None:
    """``(T, H, W, 3)`` uint8 RGB frames -> an ``mp4v`` mp4 at ``fps``.

    Written to a temporary file next to ``path`` and renamed into place, so
    ``path`` exists only once it is complete (resuming runs skip existing
    outputs).
    """
    path = Path(path)
    T, H, W, _ = frames.shape
    # cv2 picks the container from the suffix, so the temporary name keeps it.
    partial = path.with_name(f".{path.stem}.partial{path.suffix}")
    writer = cv2.VideoWriter(str(partial), cv2.VideoWriter.fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for {partial} (is mp4v available?)")
    try:
        for t in range(T):
            writer.write(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"the video writer produced no data for {path}")
    os.replace(partial, path)
