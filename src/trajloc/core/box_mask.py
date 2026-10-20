"""Bounding-box loss masks on the latent grid.

The training loss blends the plain flow-matching MSE with the same MSE
restricted to the tracked objects' boxes (weight ``box_loss_weight``, 0.5 in
the released configs), which
focuses the gradient on the objects that should move. This module turns
per-frame pixel boxes into the ``(T_lat, H_lat, W_lat)`` 0/1 mask. It has two
styles, chosen by the config's ``box_mask``
(:class:`~trajloc.core.config.BoxMaskSpec`):

- ``"padded"`` (:meth:`BoxLossMask.mask_padded`, CogVideoX): each box
  also covers ``temporal_padding`` frames before and after it, pixel -> latent
  mapping by the grid ratio, and an all-ones mask when there are no boxes.
- ``"tight"`` (:meth:`BoxLossMask.mask_tight`, Wan): each box
  marks only the latent frame of its own frame, pixel -> latent mapping by the
  VAE stride of 8, no fallback.
"""

from __future__ import annotations

import torch

from trajloc.core.config import BoxMaskStyle
from trajloc.core.latent_grid import VAE_SPATIAL_STRIDE, VAE_TEMPORAL_STRIDE

Box = tuple[float, float, float, float]  # (x, y, w, h) in pixels


class BoxLossMask:
    """Per-clip 3D box masks at the latent grid resolution.

    Args:
        latent_size: ``(H_lat, W_lat)`` of the diffusion-target grid.
        temporal_padding_frames: video frames each box is extended by, before
            and after (``"padded"`` style only).
        temporal_compression: VAE temporal stride (4 for both backbones).
    """

    def __init__(
        self,
        latent_size: tuple[int, int],
        temporal_padding_frames: int,
        temporal_compression: int = VAE_TEMPORAL_STRIDE,
    ) -> None:
        if latent_size[0] <= 0 or latent_size[1] <= 0:
            raise ValueError(f"latent_size must be positive, got {latent_size}")
        if temporal_padding_frames < 0:
            raise ValueError(f"temporal_padding_frames must be >= 0, got {temporal_padding_frames}")

        self.latent_h, self.latent_w = latent_size
        self.temporal_padding = temporal_padding_frames
        self.temporal_compression = temporal_compression

    def mask_tight(
        self,
        frame_boxes: list[list[Box]],
        num_frames: int,
    ) -> torch.Tensor:
        """``"tight"`` style: video frame ``t`` maps to latent frame
        ``int(t * (T_lat - 1) / (T - 1))``; boxes map to latent cells by the
        VAE stride of 8. No temporal padding, no empty-mask fallback."""
        T = num_frames
        T_lat = (T - 1) // self.temporal_compression + 1
        H_lat = self.latent_h
        W_lat = self.latent_w

        mask = torch.zeros(T_lat, H_lat, W_lat)
        for t, boxes in enumerate(frame_boxes):
            t_lat = int(t * (T_lat - 1) / max(T - 1, 1))
            for box in boxes:
                if box is None:
                    continue
                bx, by, bw, bh = box
                lx1 = max(0, int(bx / VAE_SPATIAL_STRIDE))
                ly1 = max(0, int(by / VAE_SPATIAL_STRIDE))
                lx2 = min(W_lat, int((bx + bw) / VAE_SPATIAL_STRIDE) + 1)
                ly2 = min(H_lat, int((by + bh) / VAE_SPATIAL_STRIDE) + 1)
                mask[t_lat, ly1:ly2, lx1:lx2] = 1.0
        return mask

    def mask_padded(
        self,
        frame_boxes: list[list[Box]],
        video_size: tuple[int, int],
    ) -> torch.Tensor:
        """``"padded"`` style. ``frame_boxes[t]`` holds the pixel boxes
        visible at video frame ``t``. Boxes fully outside the frame are
        ignored; with none left, the mask is all ones (plain MSE)."""
        num_frames = len(frame_boxes)
        video_h, video_w = video_size

        num_latent_frames = (num_frames - 1) // self.temporal_compression + 1
        mask = torch.zeros(num_latent_frames, self.latent_h, self.latent_w)

        has_any_box = False

        for frame_idx, boxes in enumerate(frame_boxes):
            for box in boxes:
                if box is None:
                    continue
                bx, by, bw, bh = box
                if bx + bw <= 0 or by + bh <= 0:
                    continue
                if bx >= video_w or by >= video_h:
                    continue

                has_any_box = True

                start_frame = max(0, frame_idx - self.temporal_padding)
                end_frame = min(num_frames, frame_idx + self.temporal_padding + 1)
                start_lat = start_frame // self.temporal_compression
                end_lat = min(
                    (end_frame - 1) // self.temporal_compression + 1,
                    num_latent_frames,
                )

                bx_clamped = max(0, bx)
                by_clamped = max(0, by)
                bx2_clamped = min(bx + bw, video_w)
                by2_clamped = min(by + bh, video_h)

                lat_x = int(bx_clamped * self.latent_w / video_w)
                lat_y = int(by_clamped * self.latent_h / video_h)
                lat_x2 = max(lat_x + 1, int(bx2_clamped * self.latent_w / video_w))
                lat_y2 = max(lat_y + 1, int(by2_clamped * self.latent_h / video_h))

                lat_x = max(0, min(lat_x, self.latent_w - 1))
                lat_y = max(0, min(lat_y, self.latent_h - 1))
                lat_x2 = min(lat_x2, self.latent_w)
                lat_y2 = min(lat_y2, self.latent_h)

                mask[start_lat:end_lat, lat_y:lat_y2, lat_x:lat_x2] = 1.0

        if not has_any_box or mask.sum() == 0:
            mask.fill_(1.0)

        return mask

    def mask(
        self,
        style: BoxMaskStyle,
        frame_boxes: list[list[Box]],
        *,
        num_frames: int,
        video_size: tuple[int, int],
    ) -> torch.Tensor:
        """The mask in ``style`` (see :class:`BoxMaskStyle`)."""
        if style is BoxMaskStyle.TIGHT:
            return self.mask_tight(frame_boxes, num_frames=num_frames)
        if style is BoxMaskStyle.PADDED:
            return self.mask_padded(frame_boxes, video_size)
        raise ValueError(f"unknown box mask style {style!r}")
