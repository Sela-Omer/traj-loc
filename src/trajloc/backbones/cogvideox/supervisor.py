"""Attention supervision for CogVideoX-5B: swaps each block's attention processor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from trajloc.backbones.cogvideox.attention import SupervisedJointAttnProcessor
from trajloc.backbones.cogvideox.constants import NUM_BLOCKS
from trajloc.core.supervision import AttentionSupervisor, SupervisedTokens


class CogVideoXSupervisor(AttentionSupervisor):
    """Supervises ``transformer.transformer_blocks[i].attn1`` of CogVideoX's 42 blocks.

    Patching sets a :class:`SupervisedJointAttnProcessor` on the block's
    attention (diffusers' ``set_processor``); restoring puts the original
    processor back. σ is trainable.
    """

    num_blocks = NUM_BLOCKS
    trainable_sigma = True

    def _blocks(self, model: Any) -> Sequence[nn.Module]:
        blocks: Sequence[nn.Module] = model.transformer_blocks
        return blocks

    def _patch_block(self, block: Any, tokens: list[SupervisedTokens], heatmaps: torch.Tensor):
        original = block.attn1.get_processor()
        block.attn1.set_processor(
            SupervisedJointAttnProcessor(
                per_sample_positions=[sample.positions for sample in tokens],
                target_heatmaps=heatmaps,
            )
        )
        return original

    def _unpatch_block(self, block: Any, saved) -> None:
        block.attn1.set_processor(saved)
