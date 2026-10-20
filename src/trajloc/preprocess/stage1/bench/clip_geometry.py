"""Per-clip geometry of the benchmark builders.

A clip is the source video's first 49 frames (``picks``), center-cropped to
3:2 and resized to 480x720 (HxW) (``crop``, source pixels). Masks are bool at
clip size; tracks come back from the tracker in source pixels and
are projected to clip pixels in float64. (SA-V's builder uses uint8 masks
and float32 tracks, ``sav/extract.py``: each builder keeps its own
convention.)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from trajloc.datasets.layout import CLIP_HEIGHT, CLIP_NUM_FRAMES, CLIP_WIDTH
from trajloc.preprocess.stage1.geometry import CropWindow
from trajloc.preprocess.stage1.swarm import SWARM_SIZE, sample_median_neighborhood
from trajloc.preprocess.stage1.tracking import QueryRange, Trajectory, anchor_trajectory


def clip_source_frames(n_src: int) -> list[int]:
    """The source frames of the clip: the first ``CLIP_NUM_FRAMES``, as they
    are (no temporal resampling), so the benchmark's motion is unchanged and
    its tracks index 1:1 (MoveBench: 81 frames at 16 fps; MagicBench: 49).
    A shorter video repeats its last frame."""
    if n_src <= 0:
        return []
    return [min(i, n_src - 1) for i in range(CLIP_NUM_FRAMES)]


def write_sam2_frames(
    source_frames: list[np.ndarray],
    picks: list[int],
    crop: CropWindow,
    out_frames_dir: Path,
) -> None:
    """Write the clip's frames as ``00000.jpg`` onwards (the input of
    SAM 2's video predictor); existing files are kept."""
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    for new_idx, src_idx in enumerate(picks):
        dst = out_frames_dir / f"{new_idx:05d}.jpg"
        if dst.exists():
            continue
        bgr_source = source_frames[min(src_idx, len(source_frames) - 1)]
        bgr_clip = crop.warp(bgr_source)
        cv2.imwrite(str(dst), bgr_clip)


def ground_truth_trajectory(
    tracks: np.ndarray, visibility: np.ndarray, obj_col: int, picks: list[int], crop: CropWindow
) -> Trajectory | None:
    """Object ``obj_col``'s ground-truth track in clip pixels, valid where
    the benchmark marks it visible and it is finite and in the frame;
    ``None`` if it is never valid."""
    T_src = tracks.shape[0]
    rx = np.full(CLIP_NUM_FRAMES, np.nan, dtype=np.float64)
    ry = np.full(CLIP_NUM_FRAMES, np.nan, dtype=np.float64)
    valid = np.zeros(CLIP_NUM_FRAMES, dtype=bool)
    for t, src_idx in enumerate(picks):
        s = min(src_idx, T_src - 1)
        if not bool(visibility[s, obj_col]):
            continue
        nx, ny = float(tracks[s, obj_col, 0]), float(tracks[s, obj_col, 1])
        if not (np.isfinite(nx) and np.isfinite(ny)):
            continue
        clip_x, clip_y = crop.to_clip_pixels(nx, ny, dtype=np.float64)
        px, py = float(clip_x), float(clip_y)
        if not (0 <= px < CLIP_WIDTH and 0 <= py < CLIP_HEIGHT):
            continue
        rx[t], ry[t], valid[t] = px, py, True
    if not valid.any():
        return None
    return Trajectory(rx, ry, valid)


def seed_swarm(
    masks: np.ndarray, crop: CropWindow, H_source: int, W_source: int
) -> tuple[int, np.ndarray] | None:
    """A swarm for the object: on the frame where its mask is largest, points
    around the mask's median (:func:`~..swarm.sample_median_neighborhood`).

    Returns ``(clip frame index, (n, 2) native-pixel points)``, or ``None``
    if the mask is always empty."""
    best_t_idx: int | None = None
    best_area = 0
    best_m_in_crop: np.ndarray | None = None
    for t_idx in range(CLIP_NUM_FRAMES):
        frame_mask = masks[t_idx]
        if not frame_mask.any():
            continue
        # Zero outside the crop, so every seed lands in the clip frame.
        m_in_crop = crop.to_source_mask(frame_mask, H_source, W_source)
        area = int(m_in_crop.sum())
        if area > best_area:
            best_area, best_t_idx, best_m_in_crop = area, t_idx, m_in_crop
    if best_t_idx is None or best_m_in_crop is None:
        return None
    pts = sample_median_neighborhood(best_m_in_crop, SWARM_SIZE)
    if pts.shape[0] == 0:
        return None
    return best_t_idx, pts


def object_trajectory(
    object_masks: np.ndarray,
    query_idx: int,
    ranges: list[QueryRange],
    tracks: np.ndarray,
    picks: list[int],
    crop: CropWindow,
) -> Trajectory | None:
    """An object's trajectory from a tracker pass: its swarm (entry
    ``query_idx`` of ``ranges``) projected to clip pixels, then
    :func:`~trajloc.preprocess.stage1.tracking.anchor_trajectory`. ``None``
    if the object had no swarm or no usable anchor."""
    if query_idx < 0:
        return None
    r = ranges[query_idx]
    if r.start == r.end:
        return None
    tracks_clip_source = tracks[list(picks), r.start : r.end, :]
    sx_clip, sy_clip = crop.to_clip_pixels(
        tracks_clip_source[..., 0], tracks_clip_source[..., 1], dtype=np.float64
    )
    return anchor_trajectory(sx_clip, sy_clip, [object_masks[t] for t in range(CLIP_NUM_FRAMES)])


CROP_PAD_PX = 100  # context around the object's box in its naming crop
MAGENTA_RGB = (255, 0, 255)  # the naming crop's outline color


def highlight_crop(
    rgb: np.ndarray,
    mask: np.ndarray,
    pad_px: int = CROP_PAD_PX,
    dim: float = 0.25,
    contour: tuple[int, int, int] = MAGENTA_RGB,
    contour_px: int = 3,
) -> Image.Image | None:
    """The crop Qwen names an object from: the mask's box plus ``pad_px`` on
    each side, with the object at full brightness and a magenta outline, and
    every other pixel (other objects included) dimmed and desaturated, so a
    salient neighbor can't take the label. The fixed pad gives every object
    the same context.

    Args:
        mask: (H, W) bool — the target object's mask at its max-area frame.
        pad_px: absolute pixel buffer on each side of the mask's box.
        dim: brightness/saturation multiplier for the pixels outside the target.
        contour / contour_px: target boundary outline.
    Returns a PIL.Image (RGB) or None if the mask is empty.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    H, W = mask.shape
    x0 = max(0, int(xs.min()) - pad_px)
    y0 = max(0, int(ys.min()) - pad_px)
    x1 = min(W, int(xs.max()) + pad_px + 1)
    y1 = min(H, int(ys.max()) + pad_px + 1)
    sub_rgb = rgb[y0:y1, x0:x1].copy()
    sub_mask = mask[y0:y1, x0:x1].astype(bool)
    if sub_rgb.size == 0 or not sub_mask.any():
        return None

    out = sub_rgb.astype(np.float32)
    gray_lum = out.mean(axis=2, keepdims=True)
    desat = 0.5 * out + 0.5 * gray_lum  # pull non-target toward grayscale
    dimmed = (desat * dim).astype(np.float32)
    # Everything outside the object's mask, other objects included.
    outside = ~sub_mask
    out[outside] = dimmed[outside]
    out = np.clip(out, 0, 255).astype(np.uint8)

    m_u8 = sub_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, tuple(int(c) for c in contour), contour_px)
    return Image.fromarray(out)
