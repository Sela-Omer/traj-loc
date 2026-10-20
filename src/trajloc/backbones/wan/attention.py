"""Supervised cross-attention for Wan (DiffSynth's ``CrossAttention``).

Wan's text conditioning is cross-attention from the video tokens to
``y = [CLIP image tokens (257), T5 text tokens]``. The supervised forward
leaves the image branch as it is and computes the text branch with the
attention matrix materialized per head: the columns of the supervised text
tokens are overwritten with their objects' heatmaps, and each row is
renormalized to sum to 1 (column replacement): at video position ``q``,
token ``i`` gets weight ``heatmap_i[q] / Z[q]``, ``Z[q]`` the row's sum after
the overwrite.

The attention weights are built under ``no_grad``, so Q and K get no gradient
through this branch (one head at a time keeps memory low). V keeps its
gradient, which is how the LoRA-adapted value projection learns.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

from trajloc.backbones.wan.constants import IMAGE_TOKEN_COUNT


def supervised_cross_attention_forward(
    original_module,
    per_sample_positions: list[list[int]],
    target_heatmaps: torch.Tensor,
) -> Callable:
    """Construct a supervised forward bound to ``original_module``.

    Args:
        original_module: the block's DiffSynth ``CrossAttention`` (only its
            ``num_heads`` / ``head_dim`` are read).
        per_sample_positions: per sample, the text positions whose attention
            column is replaced.
        target_heatmaps: ``(B, N_sup, T_lat * H_lat * W_lat)`` flattened
            heatmaps aligned with the positions. Used detached: they are
            targets, not trained.

    Returns:
        ``forward(self_module, x, y)``, to be bound with ``types.MethodType``.
    """
    batched_mode = len(per_sample_positions) > 1
    positions = per_sample_positions[0]  # the BS=1 path

    flat_heatmaps = target_heatmaps  # (B, N_sup, video_tokens)
    num_heads = original_module.num_heads
    head_dim = original_module.head_dim

    def supervised_forward(self_module, x, y):
        b = x.shape[0]
        n = num_heads
        d = head_dim

        if self_module.has_image_input:
            image_tokens = y[:, :IMAGE_TOKEN_COUNT]
            text_tokens = y[:, IMAGE_TOKEN_COUNT:]
        else:
            text_tokens = y
            image_tokens = None

        # One query for both the image and the text branch.
        q_full = self_module.norm_q(self_module.q(x))  # (b, L_video, dim)

        # Image (CLIP) branch: unchanged.
        image_out = None
        if self_module.has_image_input and image_tokens is not None:
            image_keys = self_module.norm_k_img(self_module.k_img(image_tokens))
            image_values = self_module.v_img(image_tokens)
            from diffsynth.models.wan_video_dit import flash_attention as dit_flash_attention

            image_out = dit_flash_attention(q_full, image_keys, image_values, num_heads=n)

        # Text branch: supervised, exact column replacement.
        text_keys = self_module.norm_k(self_module.k(text_tokens))
        text_values = self_module.v(text_tokens)

        L_video = q_full.shape[1]
        L_text = text_keys.shape[1]
        scale = 1.0 / math.sqrt(d)

        q_heads = q_full.view(b, L_video, n, d).transpose(1, 2)
        k_heads = text_keys.view(b, L_text, n, d).transpose(1, 2)
        v_heads = text_values.view(b, L_text, n, d).transpose(1, 2)

        heatmaps = flat_heatmaps.to(q_heads.device, q_heads.dtype).detach()
        if heatmaps.shape[0] != b:
            heatmaps = heatmaps.expand(b, -1, -1)

        # Process heads sequentially to keep peak memory low.
        text_output_heads = []
        for h in range(n):
            with torch.no_grad():
                q_h = q_heads[:, h]
                k_h = k_heads[:, h]

                logits = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
                attn_weights = F.softmax(logits.float(), dim=-1).to(q_h.dtype)
                del logits

                attn_weights = attn_weights.clone()
                if not batched_mode:
                    for i, pos in enumerate(positions):
                        attn_weights[:, :, pos] = heatmaps[:, i, :]
                else:
                    for sample in range(b):
                        for i, pos in enumerate(per_sample_positions[sample]):
                            attn_weights[sample, :, pos] = heatmaps[sample, i, :]

                row_sums = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                attn_weights = attn_weights / row_sums

            # Outside no_grad — V flows gradients through LoRA.
            v_h = v_heads[:, h]
            text_output_heads.append(torch.matmul(attn_weights, v_h))
            del attn_weights

        text_out = torch.stack(text_output_heads, dim=1).transpose(1, 2).reshape(b, L_video, n * d)

        x_out = text_out if image_out is None else (text_out + image_out)
        return self_module.o(x_out)

    return supervised_forward
