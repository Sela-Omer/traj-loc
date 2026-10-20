"""CogVideoX's training loss: v-prediction with a bounding-box term."""

from __future__ import annotations

from typing import Any

import torch


def box_weighted_velocity_loss(
    scheduler: Any,
    model_output: torch.Tensor,
    latent: torch.Tensor,
    latent_noisy: torch.Tensor,
    timesteps: torch.Tensor,
    box_masks: torch.Tensor,
    *,
    box_loss_weight: float,
    mask_dtype: torch.dtype,
) -> torch.Tensor:
    """The v-prediction MSE weighted by ``1 / (1 - alpha_bar_t)``, blended
    ``(1 - w) * mse + w * box_mse`` with ``w = box_loss_weight``.

    ``box_mse`` sums the error over the latent channels and divides by the
    number of masked ``(F, H, W)`` cells, so it is ``C`` (16) times a
    per-element masked mean; Wan's is a per-element mean. For a batch of more
    than one it is the mean of the per-sample terms (so each sample weighs the
    same however large its boxes are). Each backbone keeps its own
    (:mod:`trajloc.backbones`).

    Args:
        scheduler: the diffusers scheduler (``get_velocity``, ``alphas_cumprod``).
        model_output, latent, latent_noisy: ``(B, F, C, H, W)``.
        timesteps: ``(B,)``.
        box_masks: ``(B, F, H, W)`` 0/1.
        box_loss_weight: ``w``.
        mask_dtype: the dtype the mask is cast to (the training weight dtype).
    """
    batch_size = latent.shape[0]
    latent_pred = scheduler.get_velocity(model_output, latent_noisy, timesteps)
    alphas_cumprod = scheduler.alphas_cumprod[timesteps]
    weights = 1 / (1 - alphas_cumprod)
    while len(weights.shape) < len(latent_pred.shape):
        weights = weights.unsqueeze(-1)

    latent_loss = weights * (latent_pred - latent) ** 2  # (B, F, C, H, W)
    per_sample_base = latent_loss.reshape(batch_size, -1).mean(dim=1)  # (B,)
    loss = per_sample_base.mean()

    mask = box_masks.to(latent_loss.device, dtype=mask_dtype).unsqueeze(2)
    if mask.shape[:2] != latent_loss.shape[:2] or mask.shape[3:] != latent_loss.shape[3:]:
        raise ValueError(
            f"box masks {tuple(box_masks.shape)} don't match the latents "
            f"{tuple(latent_loss.shape)} (expected (B, F, H, W))"
        )

    if batch_size == 1:
        box_loss = (latent_loss * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        per_sample = [
            (latent_loss[b : b + 1] * mask[b : b + 1]).sum() / mask[b : b + 1].sum().clamp(min=1.0)
            for b in range(batch_size)
        ]
        box_loss = torch.stack(per_sample).mean()
    return (1 - box_loss_weight) * loss + box_loss_weight * box_loss
