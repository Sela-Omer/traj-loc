"""Attention supervision: make each object's text tokens attend where it is.

TrajLoc supervises the text-to-video attention of the backbone's transformer
blocks. For every supervised text token (one per object), its attention
weights over the video tokens are *replaced* by that object's trajectory
heatmap, so the token's value vector is written exactly where the object
should be in every frame.

:class:`AttentionSupervisor` is the backbone-independent part, a Template
Method. :meth:`AttentionSupervisor.install` resolves which text positions to
supervise, validates them, builds the heatmaps and patches every configured
block through the backbone's hooks (:meth:`~AttentionSupervisor._patch_block`
and friends): Wan swaps in a column-replacing cross-attention, CogVideoX a
joint-attention processor applying the additive correction. :meth:`restore` undoes it.

Lifecycle: call ``install`` once per forward pass. It first restores any
earlier installation. Never ``restore`` between the forward and the backward
pass: gradient checkpointing recomputes the forward during backward, and the
recomputation must see the same supervised attention.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, final

import torch
import torch.nn as nn

from trajloc.core.config import TrajLocConfig
from trajloc.core.heatmap import HeatmapGenerator
from trajloc.core.latent_grid import num_latent_frames, token_grid

# Object index -> text-token position, for one sample.
TokenPositions = dict[int, int]


@dataclass(frozen=True)
class SupervisedTokens:
    """One sample's supervised text tokens.

    Attributes:
        positions: the text positions whose attention is replaced, in
            object-index order.
        object_indices: the object whose heatmap each position gets
            (aligned with ``positions``).
    """

    positions: list[int]
    object_indices: list[int]

    @classmethod
    def resolve(
        cls, trajectory_tokens: TokenPositions, noun_tokens: TokenPositions | None
    ) -> SupervisedTokens:
        """The supervised token of each object: its appearance token (its
        noun) when the sample has any, else its ``<traj_i>`` token."""
        source = noun_tokens if noun_tokens else trajectory_tokens
        sorted_items = sorted(source.items())
        return cls(
            positions=[pos for _, pos in sorted_items],
            object_indices=[object_index for object_index, _ in sorted_items],
        )


class AttentionSupervisor(nn.Module, metaclass=ABCMeta):
    """Heatmap supervision over a backbone's transformer blocks (Template Method).

    :meth:`install` and :meth:`restore` are final. Subclasses set
    :attr:`num_blocks` (the default layer set) and :attr:`trainable_sigma`,
    and implement the three hooks :meth:`_blocks`, :meth:`_patch_block` and
    :meth:`_unpatch_block`. The only
    state is the heatmap generator (``heatmap_generator.*`` in the state
    dict) and the record of what is currently patched.

    Args:
        text_seq_length: number of text tokens; supervised positions must be
            inside ``[0, text_seq_length)``.
        heatmap_sigma: heatmap σ in pixels.
        supervised_blocks: blocks to supervise; ``None`` (or empty) means all
            :attr:`num_blocks` of them, as the released models are (a subset
            is for experiments).
        pixel_height, pixel_width, grid_height, grid_width,
        num_latent_frames: the heatmap grid (see :class:`HeatmapGenerator`).
    """

    num_blocks: ClassVar[int]
    trainable_sigma: ClassVar[bool]

    def __init__(
        self,
        *,
        text_seq_length: int,
        heatmap_sigma: float,
        supervised_blocks: list[int] | None,
        pixel_height: int,
        pixel_width: int,
        grid_height: int,
        grid_width: int,
        num_latent_frames: int,
    ) -> None:
        super().__init__()
        self.supervised_blocks = supervised_blocks or list(range(self.num_blocks))
        self.text_seq_length = text_seq_length
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_latent_frames = num_latent_frames
        self.heatmap_generator = HeatmapGenerator(
            heatmap_sigma=heatmap_sigma,
            pixel_height=pixel_height,
            pixel_width=pixel_width,
            grid_height=grid_height,
            grid_width=grid_width,
            num_latent_frames=num_latent_frames,
            trainable_sigma=self.trainable_sigma,
        )
        # layer index -> what _patch_block replaced there (for restore()).
        self._patched: dict[int, Any] = {}

    @classmethod
    def from_config(cls, config: TrajLocConfig, **backbone_settings: Any) -> AttentionSupervisor:
        """The supervisor of a TrajLoc config: σ and the heatmap grid of the
        config's clip size, over all :attr:`num_blocks` blocks (TrajLoc
        localizes every block's attention). ``backbone_settings`` are the
        subclass's own arguments."""
        pixel_height, pixel_width = config.video_size
        grid_height, grid_width = token_grid(config.video_size)
        return cls(
            heatmap_sigma=config.heatmap_sigma,
            supervised_blocks=None,
            pixel_height=pixel_height,
            pixel_width=pixel_width,
            grid_height=grid_height,
            grid_width=grid_width,
            num_latent_frames=num_latent_frames(config.num_frames),
            **backbone_settings,
        )

    @property
    def log_sigma(self) -> torch.Tensor:
        """log σ (a parameter or a buffer, per :attr:`trainable_sigma`)."""
        return self.heatmap_generator.log_sigma

    @property
    def sigma(self) -> torch.Tensor:
        """σ in pixels."""
        return self.heatmap_generator.sigma

    @property
    def is_installed(self) -> bool:
        """Whether any block is currently patched."""
        return bool(self._patched)

    # ---------------------------------------------------------------- template

    @final
    def install(
        self,
        model: nn.Module,
        trajectory_coords: torch.Tensor,
        trajectory_token_positions: list[TokenPositions],
        noun_token_positions: list[TokenPositions] | None = None,
    ) -> None:
        """Supervise ``model``'s configured blocks for this batch.

        Args:
            model: the backbone transformer (see the subclass).
            trajectory_coords: ``(B, N, T_video, 2)`` in [0, 1], -1 padded.
            trajectory_token_positions: each object's ``<traj_i>`` token
                position, one dict per sample (``B`` dicts).
            noun_token_positions: each object's appearance-token position,
                one dict per sample, or ``None``.
        """
        if self._patched:
            self.restore(model)

        tokens = self._supervised_tokens(
            trajectory_coords.shape[0], trajectory_token_positions, noun_token_positions
        )
        self._validate(tokens)
        heatmaps = self._padded_heatmaps(trajectory_coords, tokens)
        B, max_supervised = heatmaps.shape[:2]
        heatmaps = heatmaps.reshape(B, max_supervised, -1)  # video-token order: (t, h, w)

        blocks = self._blocks(model)
        out_of_range = [i for i in self.supervised_blocks if i >= len(blocks)]
        if out_of_range:  # checked before patching anything: a failure leaves no block patched
            raise ValueError(
                f"supervised blocks {out_of_range} are out of range: "
                f"the model has {len(blocks)} blocks"
            )
        for block_index in self.supervised_blocks:
            self._patched[block_index] = self._patch_block(blocks[block_index], tokens, heatmaps)

    @staticmethod
    def _supervised_tokens(
        batch_size: int,
        trajectory_token_positions: list[TokenPositions],
        noun_token_positions: list[TokenPositions] | None,
    ) -> list[SupervisedTokens]:
        """Each sample's supervised tokens."""
        for name, positions in (
            ("trajectory_token_positions", trajectory_token_positions),
            ("noun_token_positions", noun_token_positions),
        ):
            if positions is None and name == "noun_token_positions":
                continue
            if not isinstance(positions, list) or len(positions) != batch_size:
                raise ValueError(
                    f"{name} must be a list of {batch_size} dicts (one per sample); "
                    f"got {type(positions).__name__}"
                )
        per_sample_noun_positions: list[dict[int, int] | None] = (
            list(noun_token_positions) if noun_token_positions else [None] * batch_size
        )
        return [
            SupervisedTokens.resolve(trajectory, noun)
            for trajectory, noun in zip(
                trajectory_token_positions, per_sample_noun_positions, strict=True
            )
        ]

    def _validate(self, tokens: list[SupervisedTokens]) -> None:
        """Every sample supervises at least one position, all inside the text."""
        batched = len(tokens) > 1
        for b, sample in enumerate(tokens):
            if not sample.positions:
                tag = f"sample {b}" if batched else "the sample"
                raise ValueError(
                    f"{tag}: no supervised text tokens (its token positions are empty)"
                )
            for pos in sample.positions:
                if not 0 <= pos < self.text_seq_length:
                    tag = f"sample {b}: " if batched else ""
                    raise ValueError(
                        f"{tag}supervised position {pos} outside the text range "
                        f"[0, {self.text_seq_length})"
                    )

    def _padded_heatmaps(
        self, trajectory_coords: torch.Tensor, tokens: list[SupervisedTokens]
    ) -> torch.Tensor:
        """``(B, max_supervised, T, H, W)``: each sample's supervised objects'
        heatmaps, in order, zero-padded to the batch's largest count."""
        all_heatmaps = self.heatmap_generator(trajectory_coords)
        max_supervised = max(len(sample.positions) for sample in tokens)
        padded = torch.zeros(
            trajectory_coords.shape[0],
            max_supervised,
            self.num_latent_frames,
            self.grid_height,
            self.grid_width,
            device=all_heatmaps.device,
            dtype=all_heatmaps.dtype,
        )
        for b, sample in enumerate(tokens):
            if sample.object_indices:
                padded[b, : len(sample.object_indices)] = all_heatmaps[b, sample.object_indices]
        return padded

    @final
    def restore(self, model: nn.Module) -> None:
        """Undo :meth:`install`. A no-op when nothing is installed."""
        blocks = self._blocks(model)
        for layer_idx, saved in self._patched.items():
            if layer_idx < len(blocks):
                self._unpatch_block(blocks[layer_idx], saved)
        self._patched.clear()

    # ------------------------------------------------------------------- hooks

    @abstractmethod
    def _blocks(self, model: nn.Module) -> Sequence[nn.Module]:
        """The model's transformer blocks, indexable by layer."""

    @abstractmethod
    def _patch_block(
        self, block: nn.Module, tokens: list[SupervisedTokens], heatmaps: torch.Tensor
    ) -> Any:
        """Make ``block``'s attention supervised; return what to restore.

        Args:
            tokens: each sample's supervised tokens (one entry for ``B == 1``).
            heatmaps: ``(B, max_supervised, T * H * W)``, flattened in
                video-token order; sample ``b``'s first
                ``len(tokens[b].positions)`` rows are aligned with its positions,
                the rest are zero padding.
        """

    @abstractmethod
    def _unpatch_block(self, block: nn.Module, saved: Any) -> None:
        """Undo :meth:`_patch_block` given what it returned."""
