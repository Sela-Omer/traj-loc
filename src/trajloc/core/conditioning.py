"""TrajLoc's conditioning of a backbone, for one forward pass.

Conditioning a batch is two steps, in this order:

1. **Appearance tokens.** The appearance encoder turns the frame-0 latent into
   one token per object, and each is written over the object's noun token in
   the prompt embedding (:data:`AppearanceInjection`).
2. **Attention supervision.** The supervisor installs heatmap-replaced
   attention on the backbone's transformer blocks
   (:class:`~trajloc.core.supervision.AttentionSupervisor`).

Both backbones condition through a :class:`TrajectoryConditioner`, in training
and at inference, so the two can't drift apart. The backbones differ only in
their supervisor and in how the appearance tokens are written.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn

from trajloc.core.appearance_encoder import AppearanceEncoder
from trajloc.core.supervision import AttentionSupervisor, TokenPositions

# ``(prompt_embeddings, appearance_tokens, noun_token_positions,
# num_objects) -> prompt_embeddings``: how the appearance tokens are
# written into the prompt embedding (a copy is returned). The two rules are
# :func:`~trajloc.core.appearance_encoder.inject_in_position_order` (Wan) and
# :func:`~trajloc.core.appearance_encoder.inject_in_object_order` (CogVideoX).
AppearanceInjection = Callable[
    [torch.Tensor, torch.Tensor, list[TokenPositions], torch.Tensor], torch.Tensor
]


@dataclass(frozen=True)
class TrajectoryInputs:
    """What a batch is conditioned on, besides its prompt embedding.

    Attributes:
        frame0_latents: ``(B, C, H, W)`` VAE latents of the first frame.
        trajectory_coords: ``(B, N, T, 2)`` in [0, 1], -1 padded.
        trajectory_starts: ``(B, N, 2)`` each object's first visible position.
        num_objects: ``(B,)`` real object counts.
        trajectory_token_positions: per sample, object ``i`` -> its ``<traj_i>`` token.
        noun_token_positions: per sample, object ``i`` -> its noun's first token.
    """

    frame0_latents: torch.Tensor
    trajectory_coords: torch.Tensor
    trajectory_starts: torch.Tensor
    num_objects: torch.Tensor
    trajectory_token_positions: list[TokenPositions]
    noun_token_positions: list[TokenPositions]


class TrajectoryConditioner:
    """Applies TrajLoc's conditioning to a backbone's forward pass.

    It holds the trainable modules; it is not an ``nn.Module`` itself, so the
    modules keep their places in their owner's state dict and parameter order.

    Args:
        supervisor: the backbone's attention supervisor.
        appearance_encoder: the appearance encoder.
        inject: how the appearance tokens are written (:data:`AppearanceInjection`).
    """

    def __init__(
        self,
        supervisor: AttentionSupervisor,
        appearance_encoder: AppearanceEncoder,
        *,
        inject: AppearanceInjection,
    ) -> None:
        self.supervisor = supervisor
        self.appearance_encoder = appearance_encoder
        self.inject = inject

    def condition(
        self, transformer: nn.Module, prompt_embedding: torch.Tensor, inputs: TrajectoryInputs
    ) -> torch.Tensor:
        """Write the appearance tokens into ``prompt_embedding`` (a copy is
        returned), then install the supervision on ``transformer``, replacing
        any earlier installation. It stays installed until :meth:`restore`
        (see :mod:`trajloc.core.supervision` for why training never restores).
        """
        appearance_tokens = self.appearance_encoder(
            inputs.frame0_latents,
            inputs.trajectory_starts,
            num_objects=inputs.num_objects,
        )
        prompt_embedding = self.inject(
            prompt_embedding,
            appearance_tokens,
            inputs.noun_token_positions,
            inputs.num_objects,
        )
        self.supervisor.install(
            transformer,
            inputs.trajectory_coords,
            inputs.trajectory_token_positions,
            inputs.noun_token_positions,
        )
        return prompt_embedding

    def restore(self, transformer: nn.Module) -> None:
        """Remove the supervision from ``transformer`` (a no-op if none)."""
        self.supervisor.restore(transformer)

    @contextmanager
    def conditioned(
        self, transformer: nn.Module, prompt_embedding: torch.Tensor, inputs: TrajectoryInputs
    ) -> Iterator[torch.Tensor]:
        """:meth:`condition` for the duration of a ``with`` block, which gets
        the conditioned prompt embedding; restored on exit (inference)."""
        prompt_embedding = self.condition(transformer, prompt_embedding, inputs)
        try:
            yield prompt_embedding
        finally:
            self.restore(transformer)

    def to(self, device: torch.device | str) -> TrajectoryConditioner:
        """Move the modules to ``device``; returns ``self``."""
        self.appearance_encoder.to(device)
        self.supervisor.to(device)
        return self

    def eval(self) -> TrajectoryConditioner:
        """Put the modules in eval mode; returns ``self``."""
        self.appearance_encoder.eval()
        self.supervisor.eval()
        return self
