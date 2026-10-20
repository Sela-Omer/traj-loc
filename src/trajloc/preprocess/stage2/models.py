"""The released weights Stage 2 loads (from ``<project>/checkpoints/``)."""

from __future__ import annotations

from pathlib import Path

from trajloc.core.trajectory_encoder import TrajectoryTokenEncoder, load_trajectory_token_encoder
from trajloc.paths import project_root

TRAJECTORY_ENCODER_FRAMES = 89  # the encoder's frame count; trajectories are resampled to it
TRAJECTORY_ENCODER_CHECKPOINT = project_root() / "checkpoints/trajectory_encoder.safetensors"


def load_released_trajectory_encoder(
    token_std: float, device: str = "cuda:0", path: Path = TRAJECTORY_ENCODER_CHECKPOINT
) -> TrajectoryTokenEncoder:
    """The released trajectory encoder (its weights at ``path``) on ``device``,
    its tokens rescaled to ``token_std``: ``encoder(coords, depths) -> (B, N, 4096)``."""
    return load_trajectory_token_encoder(
        path,
        token_std=token_std,
        num_frames=TRAJECTORY_ENCODER_FRAMES,
        device=device,
    )
