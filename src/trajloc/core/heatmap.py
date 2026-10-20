"""Gaussian trajectory heatmaps on the backbone's latent token grid.

For every trajectory and video frame, a Gaussian of width σ (the config's
``heatmap_sigma``, 30 px; CogVideoX learns it from there) is centred on the
object's position, bilinearly downsampled to the latent grid
and normalized to sum to 1. The video-frame axis is then linearly resampled
to the latent-frame axis (49 -> 13 at 4x temporal compression) and
renormalized. These heatmaps replace the attention a trajectory's text
tokens receive, so each token attends where its object is (see
:mod:`trajloc.core.supervision`).

Shapes::

    trajectory_coords  (B, N, T_video, 2)   x, y normalized to [0, 1]; a frame
                                            is invalid iff x <= 0 AND y <= 0
                                            (the -1 padding sentinel)
    returns            (B, N, T_grid, H_grid, W_grid)
                                            valid frames sum to 1, invalid ones to 0
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapGenerator(nn.Module):
    """Trajectories -> per-latent-frame Gaussian heatmaps.

    σ is stored as ``log_sigma``: a trainable ``nn.Parameter`` for CogVideoX, a
    frozen buffer for Wan (``trainable_sigma``). Both are in the state dict.

    Args:
        heatmap_sigma: σ in pixels of the ``pixel_height x pixel_width`` frame.
        pixel_height, pixel_width: video frame size.
        grid_height, grid_width: the token grid the heatmaps live on.
        num_latent_frames: latent frames (13 for 49 video frames).
        trainable_sigma: whether σ is a parameter or a buffer.
    """

    log_sigma: torch.Tensor
    y_grid: torch.Tensor  # (pixel_height, 1) pixel rows
    x_grid: torch.Tensor  # (1, pixel_width) pixel columns

    def __init__(
        self,
        heatmap_sigma: float,
        pixel_height: int = 480,
        pixel_width: int = 720,
        grid_height: int = 30,
        grid_width: int = 45,
        num_latent_frames: int = 13,
        trainable_sigma: bool = False,
    ) -> None:
        super().__init__()
        self.pixel_height = pixel_height
        self.pixel_width = pixel_width
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_latent_frames = num_latent_frames
        self.trainable_sigma = trainable_sigma

        log_sigma = torch.tensor(math.log(heatmap_sigma))
        if trainable_sigma:
            self.log_sigma = nn.Parameter(log_sigma)
        else:
            self.register_buffer("log_sigma", log_sigma)

        y_coords = torch.arange(pixel_height, dtype=torch.float32).unsqueeze(1)
        x_coords = torch.arange(pixel_width, dtype=torch.float32).unsqueeze(0)
        # Derived from the frame size: not part of the state dict.
        self.register_buffer("y_grid", y_coords, persistent=False)
        self.register_buffer("x_grid", x_coords, persistent=False)

    @property
    def sigma(self) -> torch.Tensor:
        """The Gaussian's standard deviation, in pixels (``exp(log_sigma)``)."""
        return torch.exp(self.log_sigma)

    def forward(self, trajectory_coords: torch.Tensor) -> torch.Tensor:
        """``(B, N, T_video, 2)`` coordinates -> ``(B, N, T_grid, H, W)`` heatmaps."""
        if trajectory_coords.ndim != 4 or trajectory_coords.shape[-1] != 2:
            raise ValueError(
                f"Expected trajectory_coords shape (B, N, T, 2), got {trajectory_coords.shape}"
            )

        B, num_objects, T_video, _ = trajectory_coords.shape
        device = trajectory_coords.device
        sigma = self.sigma

        if sigma < 0.1:
            raise ValueError(f"Sigma too small ({sigma.item():.6f}) — likely log_sigma underflow")

        px = trajectory_coords[..., 0] * self.pixel_width
        py = trajectory_coords[..., 1] * self.pixel_height
        invalid_mask = (trajectory_coords[..., 0] <= 0) & (trajectory_coords[..., 1] <= 0)

        if num_objects > 0 and invalid_mask.all():
            raise ValueError("All trajectories invalid (all coords <= 0) — check coordinate data")

        px = px.unsqueeze(-1).unsqueeze(-1)
        py = py.unsqueeze(-1).unsqueeze(-1)

        y_grid = self.y_grid.to(device)
        x_grid = self.x_grid.to(device)
        sigma_sq = (sigma * sigma).clamp(min=1.0)

        # Gaussian at full pixel resolution, bilinearly downsampled to the grid.
        dist_sq = (x_grid - px) ** 2 + (y_grid - py) ** 2
        heatmap_pixel = torch.exp(-dist_sq / (2.0 * sigma_sq))

        flat = heatmap_pixel.reshape(
            B * num_objects * T_video, 1, self.pixel_height, self.pixel_width
        )
        flat_down: torch.Tensor = F.interpolate(  # annotated: torch types it Any
            flat,
            size=(self.grid_height, self.grid_width),
            mode="bilinear",
            align_corners=False,
        )
        heatmaps = flat_down.reshape(B, num_objects, T_video, self.grid_height, self.grid_width)

        hm_sum = heatmaps.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        heatmaps = heatmaps / hm_sum

        invalid_expanded = invalid_mask.unsqueeze(-1).unsqueeze(-1)
        heatmaps = heatmaps * (~invalid_expanded).float()

        # Video frames -> latent frames. A latent frame is invalid when at
        # least half of its interpolated source frames are.
        T_grid = self.num_latent_frames
        if T_video != T_grid:
            H_g, W_g = self.grid_height, self.grid_width
            flat_t = heatmaps.reshape(B * num_objects, T_video, H_g * W_g).permute(0, 2, 1)
            flat_t = F.interpolate(flat_t, size=T_grid, mode="linear", align_corners=False)
            heatmaps = flat_t.permute(0, 2, 1).reshape(B, num_objects, T_grid, H_g, W_g)

            hm_sum = heatmaps.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
            heatmaps = heatmaps / hm_sum

            invalid_float = invalid_mask.float()
            invalid_down = F.interpolate(
                invalid_float.reshape(B * num_objects, 1, T_video),
                size=T_grid,
                mode="linear",
                align_corners=False,
            ).reshape(B, num_objects, T_grid)
            invalid_latent = invalid_down >= 0.5
            heatmaps = heatmaps * (~invalid_latent).unsqueeze(-1).unsqueeze(-1).float()

        if torch.isnan(heatmaps).any():
            raise RuntimeError("NaN detected in heatmaps")

        return heatmaps
