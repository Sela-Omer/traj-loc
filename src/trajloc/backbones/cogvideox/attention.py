"""Supervised joint attention for CogVideoX: the paper's additive correction.

CogVideoX runs self-attention over the concatenated ``[text (226), video]``
sequence. At 480x720x49 the full attention matrix is far too large to
materialize, so the column replacement is applied in closed form. For a
video query ``q``, let ``H[q] = sum_i heatmap_i[q]`` be the heatmap mass of
the supervised text tokens. Their attention weights become exactly
``heatmap_i[q]``, and every other key's weight is rescaled by ``1 - H[q]`` so
the row still sums to 1 (Wan instead renormalizes the overwritten row,
:mod:`trajloc.backbones.wan.attention`)::

    O[q] = (1 - H[q]) * SDPA(q, K_unsup, V_unsup) + sum_i heatmap_i[q] * V_i

That is two SDPA calls (the full one, whose video rows are overwritten, and
the one over the unsupervised keys) plus a heatmap-weighted sum of the
supervised values. Text rows are left as they are.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class SupervisedJointAttnProcessor:
    """Diffusers attention processor for CogVideoX's joint attention.

    Same call signature as the processor it replaces
    (``__call__(attn, hidden_states, encoder_hidden_states, attention_mask,
    image_rotary_emb)``), same projections, norms and RoPE (video positions
    only). Only the attention product differs.

    Args:
        per_sample_positions: per sample, the supervised text positions.
        target_heatmaps: ``(B, N_sup, T_lat * H * W)`` flattened heatmaps,
            aligned with the positions.

    The text prefix length is read from ``encoder_hidden_states`` at call time.
    """

    def __init__(
        self,
        per_sample_positions: list[list[int]],
        target_heatmaps: torch.Tensor,
    ) -> None:
        self.per_sample_positions = per_sample_positions
        self._batched_mode = len(per_sample_positions) > 1
        self._flat_heatmaps = target_heatmaps

    # --------------------------------------------------------- diffusers entry

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple | None = None,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``attn``'s joint attention over ``[text, video]``, with the
        supervised decomposition; returns ``(video, text)`` hidden states."""
        text_seq_length = encoder_hidden_states.size(1)

        # Concatenate text + video to form the joint sequence.
        joint = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, sequence_length, _ = joint.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(joint)
        key = attn.to_k(joint)
        value = attn.to_v(joint)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # RoPE on the video portion only.
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        joint_out = self._supervised_attention(
            query, key, value, attention_mask, text_seq_length, batch_size
        )

        joint_out = joint_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # to_out is a Sequential: linear projection then dropout.
        joint_out = attn.to_out[0](joint_out)
        joint_out = attn.to_out[1](joint_out)

        # Split back to text + video.
        encoder_hidden_states, hidden_states = joint_out.split(
            [text_seq_length, joint_out.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states

    # --------------------------------------------------- replace (two-SDPA)

    def _supervised_attention(self, query, key, value, attention_mask, text_seq_length, batch_size):
        """Full SDPA for every row, then the video rows are replaced by the
        supervised decomposition (per sample for BS>1). Q/K get gradients
        through the unsupervised SDPA, V through both terms; the heatmaps are
        detached."""
        full_out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        video_start = text_seq_length

        if not self._batched_mode:
            supervised_positions = self.per_sample_positions[0]
            video_out = self._supervised_video_rows(
                query=query,
                key=key,
                value=value,
                text_seq_length=text_seq_length,
                supervised_positions=supervised_positions,
                heatmaps=self._flat_heatmaps.to(full_out.device, full_out.dtype).detach(),
                batch_size=batch_size,
            )
            full_out = full_out.clone()
            full_out[:, :, video_start:, :] = video_out
            return full_out

        # BS>1: per-sample replacement.
        full_out = full_out.clone()
        all_heatmaps = self._flat_heatmaps.to(full_out.device, full_out.dtype).detach()
        for b in range(batch_size):
            positions_b = self.per_sample_positions[b]
            if not positions_b:
                continue
            num_supervised_b = len(positions_b)
            video_out_b = self._supervised_video_rows(
                query=query[b : b + 1],
                key=key[b : b + 1],
                value=value[b : b + 1],
                text_seq_length=text_seq_length,
                supervised_positions=positions_b,
                heatmaps=all_heatmaps[b : b + 1, :num_supervised_b, :],
                batch_size=1,
            )
            full_out[b : b + 1, :, video_start:, :] = video_out_b
        return full_out

    def _supervised_video_rows(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        text_seq_length: int,
        supervised_positions: list[int],
        heatmaps: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Video-row outputs ``(1 - H) * SDPA(Q, K_unsup, V_unsup) + heatmap @ V_sup``.

        Args:
            query, key, value: joint Q/K/V of this slice, ``(b, heads, L, d)``.
            text_seq_length: text prefix length.
            supervised_positions: this slice's supervised text positions.
            heatmaps: ``(b, len(supervised_positions), L_video)`` heatmaps.
            batch_size: ``b``.
        """
        if heatmaps.shape[0] != batch_size:
            heatmaps = heatmaps.expand(batch_size, -1, -1)

        q_video = query[:, :, text_seq_length:, :]
        total_seq = key.shape[2]
        supervised_set = set(supervised_positions)
        unsupervised_keys = torch.tensor(
            [i for i in range(total_seq) if i not in supervised_set],
            device=key.device,
            dtype=torch.long,
        )

        unsupervised_out = F.scaled_dot_product_attention(
            q_video,
            key[:, :, unsupervised_keys, :],
            value[:, :, unsupervised_keys, :],
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        supervised_values = value[:, :, supervised_positions, :]
        heatmap_weights = heatmaps.permute(0, 2, 1).unsqueeze(1)
        heatmap_mass = heatmaps.sum(dim=1).unsqueeze(1).unsqueeze(-1)

        supervised_out = torch.matmul(heatmap_weights, supervised_values)
        video_out = (1.0 - heatmap_mass) * unsupervised_out + supervised_out
        return video_out
