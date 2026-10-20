"""Runtime patch of Wan's UMT5 text encoder (used by Stage 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


def install_t5_inputs_embeds_patch(t5_model: nn.Module) -> None:
    """Let ``T5Encoder.forward`` take ``inputs_embeds`` instead of ids.

    Stage 2 replaces the ``<traj_i>`` token embeddings with trajectory
    embeddings before running the encoder, which the upstream
    ``forward(ids, mask)`` can't do. Patches the instance only. With
    ``inputs_embeds=None`` the behavior is the upstream one. Idempotent.
    """
    if getattr(t5_model, "_inputs_embeds_patched", False):
        return

    def patched_forward(self, ids, mask=None, inputs_embeds=None):
        x = inputs_embeds if inputs_embeds is not None else self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x

    import types as _types

    t5_model.forward = _types.MethodType(patched_forward, t5_model)
    t5_model._inputs_embeds_patched = True  # type: ignore[assignment]  # torch types attributes as Tensor | Module
