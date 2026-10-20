"""Video <-> latent geometry shared by both backbones.

Both VAEs compress 8x in space and 4x in time (a 49-frame, 480x720 clip is
13 x 60 x 90 latents). The transformers then group 2x2 latent cells into one
token, so the attention/heatmap grid is 13 x 30 x 45.
"""

from __future__ import annotations

VAE_SPATIAL_STRIDE = 8
VAE_TEMPORAL_STRIDE = 4
TOKEN_PATCH_SIZE = 2  # latent cells per transformer token, in each spatial direction


def num_latent_frames(num_frames: int) -> int:
    """Latent frames for ``num_frames`` video frames (the first frame is
    encoded on its own): ``(num_frames - 1) // 4 + 1``."""
    return (num_frames - 1) // VAE_TEMPORAL_STRIDE + 1


def token_grid(video_size: tuple[int, int]) -> tuple[int, int]:
    """``(height, width)`` of the transformer's token grid for ``(height,
    width)`` pixel frames (30 x 45 for 480 x 720): where the heatmaps live."""
    stride = VAE_SPATIAL_STRIDE * TOKEN_PATCH_SIZE
    return video_size[0] // stride, video_size[1] // stride
