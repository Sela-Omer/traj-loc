"""Clip geometry every Stage-1 builder shares.

Every dataset's clips are cut to the training geometry: 49 frames, 480x720 (HxW)
(3:2). A source frame is center-cropped to 3:2 (:func:`center_crop_window`), then
resized (:meth:`CropWindow.warp`). Trajectories and masks live in the
resulting clip-pixel coordinates.
"""

from __future__ import annotations

from typing import NamedTuple, overload

import cv2
import numpy as np

from trajloc.datasets.layout import CLIP_HEIGHT, CLIP_WIDTH

CLIP_ASPECT = CLIP_WIDTH / CLIP_HEIGHT


class CropWindow(NamedTuple):
    """A crop window, in source pixels, and the mapping it defines from the
    source frame to the clip frame (crop, then resize to 480x720 (HxW)). A tuple, so
    ``x0, y0, w, h = window`` works.

    The two Stage-1 builders use different conventions:
    SA-V warps uint8 masks (an empty crop gives ``None``) and keeps the track
    dtype; the benchmarks warp bool masks (an empty crop gives all ``False``)
    and project in float64.
    """

    x0: int
    y0: int
    width: int
    height: int

    def warp(self, frame: np.ndarray) -> np.ndarray:
        """The frame cropped and resized to the clip size (bilinear)."""
        x0, y0, cw, ch = self
        cropped = frame[y0 : y0 + ch, x0 : x0 + cw]
        return cv2.resize(cropped, (CLIP_WIDTH, CLIP_HEIGHT), interpolation=cv2.INTER_LINEAR)

    def warp_mask_uint8(self, mask: np.ndarray | None) -> np.ndarray | None:
        """A source mask cropped and resized (nearest) to the clip size, as
        uint8 0/1; ``None`` if the crop is empty (SA-V)."""
        if mask is None:
            return None
        x0, y0, cw, ch = self
        cropped = mask[y0 : y0 + ch, x0 : x0 + cw]
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            return None
        resized = cv2.resize(cropped, (CLIP_WIDTH, CLIP_HEIGHT), interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.uint8)

    @overload
    def warp_mask_bool(self, mask: np.ndarray) -> np.ndarray: ...
    @overload
    def warp_mask_bool(self, mask: None) -> None: ...
    def warp_mask_bool(self, mask: np.ndarray | None) -> np.ndarray | None:
        """A source mask cropped and resized (nearest) to the clip size, as
        bool; all ``False`` if the crop is empty (the benchmarks)."""
        if mask is None:
            return None
        x0, y0, cw, ch = self
        cropped = mask[y0 : y0 + ch, x0 : x0 + cw]
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            return np.zeros((CLIP_HEIGHT, CLIP_WIDTH), dtype=bool)
        resized = cv2.resize(
            cropped.astype(np.uint8), (CLIP_WIDTH, CLIP_HEIGHT), interpolation=cv2.INTER_NEAREST
        )
        return resized.astype(bool)

    def to_clip_pixels(self, x, y, *, dtype=None) -> tuple[np.ndarray, np.ndarray]:
        """Source-pixel coordinates in clip pixels, as ``dtype`` (``None``:
        the input's; the benchmarks use float64)."""
        x0, y0, cw, ch = self
        sx = (np.asarray(x, dtype=dtype) - x0) * (CLIP_WIDTH / cw)
        sy = (np.asarray(y, dtype=dtype) - y0) * (CLIP_HEIGHT / ch)
        return sx, sy

    @overload
    def mask_inside(self, mask: np.ndarray) -> np.ndarray: ...
    @overload
    def mask_inside(self, mask: None) -> None: ...
    def mask_inside(self, mask: np.ndarray | None) -> np.ndarray | None:
        """The source mask with every pixel outside the window zeroed, so each
        remaining pixel lands inside the clip frame."""
        if mask is None:
            return None
        x0, y0, cw, ch = self
        out = np.zeros_like(mask)
        out[y0 : y0 + ch, x0 : x0 + cw] = mask[y0 : y0 + ch, x0 : x0 + cw]
        return out

    @overload
    def to_source_mask(self, clip_mask: np.ndarray, H_source: int, W_source: int) -> np.ndarray: ...
    @overload
    def to_source_mask(self, clip_mask: None, H_source: int, W_source: int) -> None: ...
    def to_source_mask(
        self, clip_mask: np.ndarray | None, H_source: int, W_source: int
    ) -> np.ndarray | None:
        """A clip mask back in source pixels (uint8; zero outside the window)."""
        if clip_mask is None:
            return None
        x0, y0, cw, ch = self
        crop_mask = cv2.resize(
            clip_mask.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST
        )
        source_mask = np.zeros((H_source, W_source), dtype=np.uint8)
        source_mask[y0 : y0 + ch, x0 : x0 + cw] = crop_mask
        return source_mask


def center_crop_window(src_w: int, src_h: int) -> CropWindow:
    """The centered 3:2 crop window of a ``src_w x src_h`` frame."""
    src_aspect = src_w / src_h
    if src_aspect > CLIP_ASPECT:
        cw, ch = int(round(src_h * CLIP_ASPECT)), src_h
    else:
        cw, ch = src_w, int(round(src_w / CLIP_ASPECT))
    x0 = (src_w - cw) // 2
    y0 = (src_h - ch) // 2
    return CropWindow(x0, y0, cw, ch)
