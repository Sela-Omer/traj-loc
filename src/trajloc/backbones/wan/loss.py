"""Wan's training loss: flow matching with a bounding-box term."""

from __future__ import annotations

import torch


def box_weighted_flow_matching_loss(
    pipe,
    *,
    box_masks: torch.Tensor | None = None,
    box_loss_weight: float,
    return_timestep: bool = False,
    **inputs,
):
    """One training step's loss for a DiffSynth pipeline, with the box term.

    Draws a timestep (``torch.randint``) and noise (``torch.randn_like``),
    noises ``input_latents``, runs ``pipe.model_fn`` and returns::

        L = w(t) * ((1 - alpha) * mean(err) + alpha * sum(err * mask) / sum(mask))

    with ``err`` the per-element squared error against the scheduler's
    training target, reduced over the whole batch. ``box_masks`` is
    ``(B, T_lat, H_lat, W_lat)``; with ``None`` or ``alpha == 0`` it is plain
    MSE. This reimplements DiffSynth's ``FlowMatchSFTLoss``, which returns
    only a scalar and so can't be masked.

    Returns the loss, or ``(loss, timestep)`` with ``return_timestep``.
    Raises ``RuntimeError`` on a non-finite loss.
    """
    max_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_boundary, max_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    per_elem_loss = (noise_pred.float() - training_target.float()) ** 2

    if box_masks is not None and box_loss_weight > 0:
        mask = box_masks.to(per_elem_loss.device, dtype=per_elem_loss.dtype)
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)  # (B, 1, T, H, W)
        mask = mask.expand_as(per_elem_loss)
        mean_loss = per_elem_loss.mean()
        masked_loss = (per_elem_loss * mask).sum() / mask.sum().clamp(min=1.0)
        loss = (1.0 - box_loss_weight) * mean_loss + box_loss_weight * masked_loss
    else:
        loss = per_elem_loss.mean()

    loss = loss * pipe.scheduler.training_weight(timestep)

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss.item()}")

    if return_timestep:
        return loss, int(timestep.detach().flatten()[0].item())
    return loss
