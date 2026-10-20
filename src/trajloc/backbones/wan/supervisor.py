"""Attention supervision for Wan 2.1: patches each block's cross-attention."""

from __future__ import annotations

import types
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from trajloc.backbones.wan.attention import supervised_cross_attention_forward
from trajloc.backbones.wan.constants import NUM_BLOCKS, TEXT_SEQ_LENGTH
from trajloc.core.supervision import AttentionSupervisor, SupervisedTokens


class WanSupervisor(AttentionSupervisor):
    """Supervises ``model.blocks[i].cross_attn`` of Wan's 40 blocks.

    Patching shadows the block's ``cross_attn.forward`` with the supervised
    forward (an instance attribute); restoring deletes it, so ``forward``
    resolves to the class method again. σ is frozen.
    """

    num_blocks = NUM_BLOCKS
    trainable_sigma = False

    def __init__(self, **settings: Any) -> None:
        """:class:`AttentionSupervisor`'s arguments, except ``text_seq_length``
        (UMT5's, fixed)."""
        super().__init__(text_seq_length=TEXT_SEQ_LENGTH, **settings)

    def _blocks(self, model: Any) -> Sequence[nn.Module]:
        blocks: Sequence[nn.Module] = model.blocks
        return blocks

    def _patch_block(
        self, block: Any, tokens: list[SupervisedTokens], heatmaps: torch.Tensor
    ) -> None:
        supervised_fn = supervised_cross_attention_forward(
            original_module=block.cross_attn,
            per_sample_positions=[sample.positions for sample in tokens],
            target_heatmaps=heatmaps,
        )
        block.cross_attn.forward = types.MethodType(supervised_fn, block.cross_attn)

    def _unpatch_block(self, block: Any, saved: None) -> None:
        if "forward" in block.cross_attn.__dict__:
            del block.cross_attn.__dict__["forward"]
