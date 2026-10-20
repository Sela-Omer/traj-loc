"""A clip's Stage-1 inputs, read the way every Stage-2 step reads them.

Inputs are Stage-1 clips (see :mod:`trajloc.datasets.layout`). Videos are read
as ``(T, C, H, W)`` RGB float32 in [-1, 1], what both VAEs expect.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from trajloc.datasets.layout import CLIP_NUM_FRAMES


def load_json(path: Path) -> dict:
    """The parsed JSON file."""
    with open(path) as f:
        return json.load(f)


def read_video_tensor(path: Path) -> torch.Tensor | None:
    """Read 49 frames of a Stage-1 mp4 as ``(T=49, 3, H=480, W=720)`` float32 in
    ``[-1, 1]``. Returns None if the video is short/corrupt."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frames = []
    for _ in range(CLIP_NUM_FRAMES):
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            return None
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    arr = np.stack(frames)  # (T, H, W, 3)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 127.5 - 1.0


def trajectory_tensors(
    trajectory_objects: dict,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    """Stage-1 trajectories.json -> ``(coords (N,49,2), depths (N,49), labels, ids)``.

    Stage 1 writes depths ready to cache (0 = near, 1 = far); here the JSON
    ``None`` just becomes the tensor sentinel -1.

    ``positions[t] = None``  -> ``coords[i, t] = (-1, -1)``.
    ``depths[t]    = None``  -> ``depths[i, t] = -1``.
    """
    objects = trajectory_objects["objects"]
    sorted_ids = sorted(objects.keys(), key=lambda x: int(x))
    N = len(sorted_ids)
    T = CLIP_NUM_FRAMES

    coords = torch.full((N, T, 2), -1.0, dtype=torch.float32)
    depths = torch.full((N, T), -1.0, dtype=torch.float32)
    labels: list[str] = []
    ids: list[str] = []
    for i, oid in enumerate(sorted_ids):
        obj = objects[oid]
        positions = obj["positions"]
        depths_list = obj["depths"]
        labels.append(obj["label"])
        ids.append(oid)
        for t in range(T):
            p = positions[t] if t < len(positions) else None
            if p is not None:
                coords[i, t, 0] = float(p[0])
                coords[i, t, 1] = float(p[1])
            d = depths_list[t] if t < len(depths_list) else None
            if d is not None:
                depths[i, t] = float(d)

    return coords, depths, labels, ids


def trajectory_starts(coords: torch.Tensor) -> torch.Tensor:
    """Each trajectory's first position with ``x >= 0``; ``(0, 0)`` if none."""
    N, T, _ = coords.shape
    starts = torch.zeros(N, 2, dtype=torch.float32)
    for i in range(N):
        for t in range(T):
            if coords[i, t, 0] >= 0:
                starts[i] = coords[i, t]
                break
    return starts
