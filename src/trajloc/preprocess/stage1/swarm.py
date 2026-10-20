"""Seed points for tracking an object: a small swarm near its mask's center.

An object is tracked from the frame where its mask is largest (the seed
frame). :func:`sample_median_neighborhood` picks up to ``SWARM_SIZE`` points
inside the mask there, within ``0.4 * sqrt(area / pi)`` of the mask's
median point; the tracker then follows all of them and the best one is kept
(:func:`trajloc.preprocess.stage1.tracking.anchor_trajectory`). Staying near
the median keeps the chosen point on the object's body rather than on an
extremity. Sampling is a deterministic grid, with a seeded random fallback.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

SWARM_SIZE = 24
NEIGHBOR_RADIUS_FACTOR = 0.4
MIN_NEIGHBORS = 4  # fewer than max(this, n_target // 4) near the median: sample the whole mask
GRID_OVERSAMPLING = 1.5  # the sampling grid has this many times n_target points
FALLBACK_SEED = 0  # the random fallback's seed


def stratified_mask_samples(mask: np.ndarray, n_target: int) -> np.ndarray:
    """Up to ``n_target`` mask pixels spread evenly over the mask, as
    ``(n, 2)`` float32 ``(x, y)``: every pixel if there are at most
    ``n_target``; else :func:`_grid_sample` over the mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(xs) <= n_target:
        return np.stack([xs, ys], axis=1).astype(np.float32)

    def on_mask(x: int, y: int) -> bool:
        return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0

    return _grid_sample(xs, ys, on_mask, n_target)


def sample_median_neighborhood(
    mask: np.ndarray, n_target: int, radius_factor: float = NEIGHBOR_RADIUS_FACTOR
) -> np.ndarray:
    """Up to ``n_target`` points inside ``mask`` and within radius
    ``r = radius_factor * sqrt(area / pi)`` of the mask's median point, spread
    evenly (:func:`_grid_sample`), as ``(n, 2)`` float32 ``(x, y)``.

    ``mask`` may be a source-resolution mask already zeroed outside the crop;
    the median is over its nonzero pixels. If fewer than
    ``max(MIN_NEIGHBORS, n_target // 4)`` pixels are that close (a very irregular mask),
    samples the whole mask instead (:func:`stratified_mask_samples`).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(xs) <= n_target:
        return np.stack([xs, ys], axis=1).astype(np.float32)

    median_x = float(np.median(xs))
    median_y = float(np.median(ys))
    area = len(xs)
    radius = radius_factor * float(np.sqrt(area / np.pi))

    d2 = (xs - median_x) ** 2 + (ys - median_y) ** 2
    keep = d2 <= radius * radius
    if keep.sum() < max(MIN_NEIGHBORS, n_target // 4):
        return stratified_mask_samples(mask, n_target)

    cand_xs = xs[keep]
    cand_ys = ys[keep]
    if len(cand_xs) <= n_target:
        return np.stack([cand_xs, cand_ys], axis=1).astype(np.float32)
    candidates = set(zip(cand_xs.tolist(), cand_ys.tolist()))
    return _grid_sample(cand_xs, cand_ys, lambda x, y: (x, y) in candidates, n_target)


def _grid_sample(
    xs: np.ndarray, ys: np.ndarray, contains: Callable[[int, int], bool], n_target: int
) -> np.ndarray:
    """Up to ``n_target`` of the candidate pixels ``(xs, ys)`` (more than
    ``n_target`` of them), as ``(n, 2)`` float32 ``(x, y)``: the points of a
    regular grid over their bounding box that ``contains`` accepts, evenly
    thinned to ``n_target``; if none is accepted, a seeded random choice of
    candidates."""
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    side = int(np.ceil(np.sqrt(n_target * GRID_OVERSAMPLING)))
    gx = np.linspace(x0 + 0.5, x1 - 0.5, side)
    gy = np.linspace(y0 + 0.5, y1 - 0.5, side)
    accepted: list[tuple[float, float]] = []
    for gxi in gx:
        for gyi in gy:
            xi = int(round(gxi))
            yi = int(round(gyi))
            if contains(xi, yi):
                accepted.append((float(xi), float(yi)))
    if len(accepted) == 0:
        idx = np.random.default_rng(FALLBACK_SEED).choice(len(xs), size=n_target, replace=False)
        return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    pts = np.array(accepted, dtype=np.float32)
    if len(pts) > n_target:
        idx = np.linspace(0, len(pts) - 1, n_target).astype(int)
        pts = pts[idx]
    return pts
