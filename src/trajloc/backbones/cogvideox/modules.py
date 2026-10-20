"""TrajLoc's modules for CogVideoX, built the same way for training and inference."""

from __future__ import annotations

from trajloc.backbones.cogvideox.constants import TEXT_SEQ_LENGTH
from trajloc.backbones.cogvideox.supervisor import CogVideoXSupervisor
from trajloc.core.appearance_encoder import AppearanceEncoder, inject_in_object_order
from trajloc.core.conditioning import TrajectoryConditioner
from trajloc.core.config import TrajLocConfig


def build_conditioner(config: TrajLocConfig) -> TrajectoryConditioner:
    """The appearance encoder and the supervisor, as a conditioner that writes
    the appearance tokens in object order. The encoder is built first (its
    initialization draws from the global RNG)."""
    appearance_encoder = AppearanceEncoder.from_config(config)
    supervisor = CogVideoXSupervisor.from_config(config, text_seq_length=TEXT_SEQ_LENGTH)
    return TrajectoryConditioner(supervisor, appearance_encoder, inject=inject_in_object_order)
