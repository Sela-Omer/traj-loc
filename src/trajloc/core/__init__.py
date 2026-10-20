"""Backbone-independent building blocks (pure PyTorch).

- :class:`HeatmapGenerator`: trajectories -> Gaussian heatmaps on the latent grid.
- :class:`AttentionSupervisor`: installs heatmap-replaced attention (subclassed
  per backbone).
- :class:`AppearanceEncoder`: frame-0 latent -> one appearance token per object.
- :class:`TrajectoryConditioner`: the supervisor and the appearance encoder,
  applied to one forward pass.
- :class:`TrajectoryTokenEncoder`: the frozen trajectory encoder (Stage 2).
- :class:`BoxLossMask`: object boxes -> latent loss masks.
"""

from trajloc.core.appearance_encoder import AppearanceEncoder
from trajloc.core.box_mask import BoxLossMask, BoxMaskStyle
from trajloc.core.conditioning import TrajectoryConditioner, TrajectoryInputs
from trajloc.core.config import TrainingSet, TrajLocConfig
from trajloc.core.heatmap import HeatmapGenerator
from trajloc.core.supervision import AttentionSupervisor, SupervisedTokens
from trajloc.core.trajectory_encoder import TrajectoryTokenEncoder, load_trajectory_token_encoder

__all__ = [
    "AppearanceEncoder",
    "AttentionSupervisor",
    "BoxLossMask",
    "BoxMaskStyle",
    "HeatmapGenerator",
    "SupervisedTokens",
    "TrainingSet",
    "TrajLocConfig",
    "TrajectoryConditioner",
    "TrajectoryInputs",
    "TrajectoryTokenEncoder",
    "load_trajectory_token_encoder",
]
