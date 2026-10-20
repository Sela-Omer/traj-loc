"""Point tracking with CoTracker3, and turning a tracked swarm into a trajectory.

:class:`CoTracker` tracks many query points through a whole source
video in as few CoTracker3 passes as fit in memory: all objects' (and, for
SA-V, all clips') query points are packed into shared forward passes,
chunked by a point x frame budget, and a chunk that runs out of memory is
split and retried. The video is downscaled for tracking (long side
``TRACK_LONG_SIDE``) and tracks are returned in source pixels.

:func:`anchor_trajectory` picks, from an object's tracked swarm, the point
whose track stays inside the object's mask on the most frames, and keeps it
only on frames where it lies inside the mask's bounding box.

CoTracker3 comes from ``torch.hub``: ``cotracker3_offline`` of
``facebookresearch/co-tracker``, pinned to commit ``82e02e8``.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
import torch

from trajloc.datasets.layout import CLIP_HEIGHT, CLIP_WIDTH

logger = logging.getLogger(__name__)

TRACK_LONG_SIDE = 480  # videos are downscaled to this long side for tracking


class Trajectory(NamedTuple):
    """An object's trajectory over the clip: per frame, its position in clip
    pixels (NaN where invalid) and whether it is valid. A tuple, so
    ``x, y, valid = trajectory`` works."""

    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class QueryRange:
    """Where one query entry's points are in the returned tracks
    (``tracks[:, start:end]``); ``start == end`` if the entry had no points."""

    start: int
    end: int


class TrackResult(NamedTuple):
    """:meth:`CoTracker.run`'s tracks over all ``N`` query points."""

    tracks: np.ndarray  # (T, N, 2) float32, source pixels
    visibility: np.ndarray  # (T, N) bool
    ranges: list[QueryRange]  # per query entry, its points


class CoTracker:
    """CoTracker3 (offline) on one device; see the module docstring."""

    def __init__(
        self,
        device: str = "cuda:0",
        max_resize_long_side: int = TRACK_LONG_SIDE,
        hub_repo: str = "facebookresearch/co-tracker:82e02e8",
        hub_model: str = "cotracker3_offline",
    ) -> None:
        self.device = torch.device(device)
        self.max_resize_long_side = int(max_resize_long_side)
        # skip_validation: torch.hub otherwise accepts a commit only while it is a
        # branch or tag head upstream (and asks GitHub's API): the pinned commit is
        # loaded either way.
        model = torch.hub.load(hub_repo, hub_model, trust_repo=True, skip_validation=True)
        self.model = model.to(self.device).eval()

    def _scale_for(self, H: int, W: int) -> tuple[float, int, int]:
        """The tracking scale and frame size: the long side down to
        ``max_resize_long_side`` (never up)."""
        long_side = max(H, W)
        if long_side <= self.max_resize_long_side:
            return 1.0, H, W
        s = self.max_resize_long_side / long_side
        return s, int(round(H * s)), int(round(W * s))

    @torch.no_grad()
    def run(
        self,
        bgr_frames_src: list[np.ndarray],
        queries: Sequence[tuple[int, np.ndarray | None]],
        *,
        max_query_budget: int = 200_000,
    ) -> TrackResult | None:
        """Track every entry's query points through the whole video.

        Args:
            bgr_frames_src: the video's ``(H, W, 3)`` uint8 BGR frames.
            queries: per entry, ``(seed frame, (n, 2) source-pixel points)``.
            max_query_budget: points x frames per forward pass.

        Entries are packed, in order, into forward passes of at most
        ``max_query_budget`` points x frames each. A pass that runs out of GPU
        memory is split in half and retried; a single entry that still does
        is retried once, then left untracked (zero tracks, not visible).

        Returns the tracks of all the entries' points, or ``None`` if there
        is nothing to track.
        """
        if not queries or len(bgr_frames_src) == 0:
            return None

        T = len(bgr_frames_src)
        H_src, W_src = bgr_frames_src[0].shape[:2]
        scale, H_t, W_t = self._scale_for(H_src, W_src)
        video_t = self._video_tensor(bgr_frames_src, scale, H_t, W_t)

        ranges, entry_pts, entry_n = _layout_queries(queries, scale)
        n_total = sum(entry_n)
        if n_total == 0:
            return None

        tracks = np.zeros((T, n_total, 2), dtype=np.float32)
        vis = np.zeros((T, n_total), dtype=bool)

        buffers = (video_t, entry_pts, entry_n, ranges, tracks, vis)
        pending = deque(_pack_chunks(entry_n, T, int(max_query_budget)))
        while pending:
            chunk = pending.popleft()
            if self._run_chunk(chunk, *buffers):
                continue
            # Out of memory: split the chunk in half (both halves next, in
            # order); a single entry is retried once more, then left untracked.
            if len(chunk) <= 1:
                torch.cuda.empty_cache()
                if not self._run_chunk(chunk, *buffers):
                    logger.warning(
                        f"skipping single-entry chunk after persistent OOM "
                        f"(n_pts={sum(entry_n[ei] for ei in chunk)}, T={T})"
                    )
                continue
            mid = len(chunk) // 2
            pending.appendleft(chunk[mid:])
            pending.appendleft(chunk[:mid])

        if scale != 1.0:
            tracks = tracks / scale  # back to source pixels

        del buffers, video_t
        torch.cuda.empty_cache()
        return TrackResult(tracks, vis, ranges)

    def _video_tensor(
        self, bgr_frames: list[np.ndarray], scale: float, H_t: int, W_t: int
    ) -> torch.Tensor:
        """``(1, T, 3, H_t, W_t)`` float RGB on the device (downscaled if ``scale != 1``)."""
        if scale != 1.0:
            frames = [cv2.resize(fr, (W_t, H_t), interpolation=cv2.INTER_AREA) for fr in bgr_frames]
        else:
            frames = list(bgr_frames)
        frames_rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in frames]
        return (
            torch.from_numpy(np.stack(frames_rgb))  # (T, H, W, 3) uint8
            .permute(0, 3, 1, 2)  # (T, 3, H, W)
            .float()
            .unsqueeze(0)  # (1, T, 3, H, W)
            .to(self.device)
        )

    def _run_chunk(
        self,
        entry_idxs: list[int],
        video_t: torch.Tensor,
        entry_pts: list[tuple[int, np.ndarray] | None],
        entry_n: list[int],
        ranges: list[QueryRange],
        tracks: np.ndarray,
        vis: np.ndarray,
    ) -> bool:
        """One forward pass over the listed entries, written into ``tracks``
        and ``vis``. False if it ran out of GPU memory."""
        n_chunk = sum(entry_n[ei] for ei in entry_idxs)
        if n_chunk == 0:
            return True
        queries = np.zeros((1, n_chunk, 3), dtype=np.float32)
        offsets_in_chunk = []
        cur = 0
        for ei in entry_idxs:
            entry = entry_pts[ei]
            assert entry is not None  # _pack_chunks packs only entries with points
            seed_frame, pts_t = entry
            queries[0, cur : cur + entry_n[ei], 0] = float(seed_frame)
            queries[0, cur : cur + entry_n[ei], 1:3] = pts_t
            offsets_in_chunk.append((ei, cur, cur + entry_n[ei]))
            cur += entry_n[ei]
        try:
            queries_t = torch.from_numpy(queries).to(self.device)
            pred_tracks, pred_vis = self.model(video_t, queries=queries_t)
            tr = pred_tracks[0].detach().cpu().numpy().astype(np.float32)
            vs = pred_vis[0].detach().cpu().numpy().astype(bool)
            del pred_tracks, pred_vis, queries_t
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False
        for ei, cs, ce in offsets_in_chunk:
            r = ranges[ei]
            tracks[:, r.start : r.end, :] = tr[:, cs:ce, :]
            vis[:, r.start : r.end] = vs[:, cs:ce]
        return True


def _layout_queries(
    queries: Sequence[tuple[int, np.ndarray | None]], scale: float
) -> tuple[list[QueryRange], list[tuple[int, np.ndarray] | None], list[int]]:
    """Each entry's range in the concatenated points, its points scaled to the
    tracking resolution (``None`` if it has none), and its point count."""
    ranges: list[QueryRange] = []
    entry_pts: list[tuple[int, np.ndarray] | None] = []
    entry_n: list[int] = []
    cursor = 0
    for seed_frame, pts_src in queries:
        if pts_src is None or pts_src.shape[0] == 0:
            ranges.append(QueryRange(start=cursor, end=cursor))
            entry_pts.append(None)
            entry_n.append(0)
            continue
        pts_t = pts_src.astype(np.float32) * scale
        n = pts_t.shape[0]
        ranges.append(QueryRange(start=cursor, end=cursor + n))
        cursor += n
        entry_pts.append((int(seed_frame), pts_t))
        entry_n.append(n)
    return ranges, entry_pts, entry_n


def _pack_chunks(entry_n: list[int], T: int, budget: int) -> list[list[int]]:
    """Consecutive non-empty entries packed while points x frames fit ``budget``."""
    chunks = []
    cur: list[int] = []
    cur_n = 0
    for ei, n in enumerate(entry_n):
        if n == 0:
            continue
        if cur and (cur_n + n) * T > budget:
            chunks.append(cur)
            cur = []
            cur_n = 0
        cur.append(ei)
        cur_n += n
    if cur:
        chunks.append(cur)
    return chunks


def anchor_trajectory(
    sx_clip: np.ndarray, sy_clip: np.ndarray, masks_t: list[np.ndarray | None]
) -> Trajectory | None:
    """The trajectory of an object's best-tracked swarm point.

    Args:
        sx_clip, sy_clip: ``(T, K)`` clip-pixel positions of the ``K``
            swarm points over the clip's ``T`` frames.
        masks_t: ``T`` clip-size 0/1 masks of the object (``None``
            where it has none).

    Each point scores the number of frames where it is in the frame, finite
    and on a mask pixel (rounded). The best point (first on ties) gives the
    trajectory; a frame is valid iff the point is there in-frame, finite and
    inside the mask's bounding box. Returns ``(x, y, valid)`` (float64, NaN
    where invalid), or ``None`` if no point ever lands on the mask or no
    frame is valid.
    """
    T, K = sx_clip.shape
    scores = np.zeros(K, dtype=np.int32)
    for k in range(K):
        on_mask = 0
        for t in range(T):
            px, py = sx_clip[t, k], sy_clip[t, k]
            mask = masks_t[t]
            if not _in_clip_frame(px, py) or mask is None or not mask.any():
                continue
            ix, iy = int(round(float(px))), int(round(float(py)))
            if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix]:
                on_mask += 1
        scores[k] = on_mask

    best_k = int(np.argmax(scores))
    if int(scores[best_k]) == 0:
        return None

    rx = sx_clip[:, best_k].astype(np.float64).copy()
    ry = sy_clip[:, best_k].astype(np.float64).copy()
    valid = np.zeros(T, dtype=bool)
    for t in range(T):
        px, py = rx[t], ry[t]
        mask = masks_t[t]
        if not _in_clip_frame(px, py) or mask is None or not mask.any():
            continue
        ys, xs = np.where(mask)
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        valid[t] = x_min <= px <= x_max and y_min <= py <= y_max
    rx[~valid] = np.nan
    ry[~valid] = np.nan
    if not valid.any():
        return None
    return Trajectory(rx, ry, valid)


def _in_clip_frame(px: float, py: float) -> bool:
    """Whether a clip-pixel position is finite and inside the frame."""
    finite = bool(np.isfinite(px) and np.isfinite(py))
    return finite and 0 <= px < CLIP_WIDTH and 0 <= py < CLIP_HEIGHT
