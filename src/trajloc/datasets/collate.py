"""Batch collation for cached training clips.

A batch is a dict of tensors (and per-sample token-position lists), not a
typed record: the trainers' data loaders (accelerate's, for CogVideoX) move
mappings of tensors to the device, and the backbones' training code reads
them by key.
"""

from __future__ import annotations

from typing import Any

import torch

from trajloc.datasets.cached_clips import ClipSample


def cached_collate(batch: list[ClipSample]) -> dict[str, Any]:
    """Stack :class:`~trajloc.datasets.cached_clips.ClipSample` s into a batch.

    Per-object tensors are padded to the batch's largest object count:
    trajectory coordinates with -1 (the "invalid" sentinel), start
    positions with 0. ``num_objects`` holds the real counts; the
    appearance encoder normalizes over those only. Token-position dicts stay
    per-sample lists. ``image_embeddings`` / ``i2v_conditioning`` (Wan caches
    only) are stacked when every sample has them, and refused when only some
    samples do.
    """
    B = len(batch)

    latents = torch.stack([s.latents for s in batch])
    frame0_latents = torch.stack([s.frame0_latent for s in batch])

    # Prompt embeddings: CogVideoX caches are fixed length (226); Wan caches
    # store the actual UMT5 output without padding (variable per sample).
    # Stack directly when all match; zero-pad to max_seq otherwise.
    seqs = [s.prompt_embedding.shape[0] for s in batch]
    if len(set(seqs)) == 1:
        prompt_embeddings = torch.stack([s.prompt_embedding for s in batch])
    else:
        max_seq = max(seqs)
        dim = batch[0].prompt_embedding.shape[1]
        ref = batch[0].prompt_embedding
        prompt_embeddings = torch.zeros(B, max_seq, dim, dtype=ref.dtype)
        for i, s in enumerate(batch):
            L = s.prompt_embedding.shape[0]
            prompt_embeddings[i, :L] = s.prompt_embedding

    N_max = max(s.num_objects for s in batch)
    T = batch[0].trajectory_coords.shape[1]

    padded_coords = torch.full((B, N_max, T, 2), -1.0)
    padded_starts = torch.zeros(B, N_max, 2)
    num_objects = torch.tensor([s.num_objects for s in batch], dtype=torch.long)

    for b, s in enumerate(batch):
        n = s.num_objects
        padded_coords[b, :n] = s.trajectory_coords
        padded_starts[b, :n] = s.trajectory_starts

    noun_token_positions = [s.noun_token_positions for s in batch]
    trajectory_token_positions = [s.trajectory_token_positions for s in batch]

    box_masks = _stack_if_present(batch, "box_mask")

    out: dict[str, Any] = {
        "latents": latents,
        "frame0_latents": frame0_latents,
        "prompt_embeddings": prompt_embeddings,
        "trajectory_coords": padded_coords,
        "trajectory_starts": padded_starts,
        "num_objects": num_objects,
        "noun_token_positions": noun_token_positions,
        "trajectory_token_positions": trajectory_token_positions,
        "box_masks": box_masks,
    }

    # Optional Wan-only fields: stacked if every sample has them, absent if none
    # does (a batch mixing the two is an error).
    for key in ("image_embeddings", "i2v_conditioning"):
        stacked = _stack_if_present(batch, key)
        if stacked is not None:
            out[key] = stacked

    return out


def _stack_if_present(batch: list[ClipSample], field: str) -> torch.Tensor | None:
    """``field`` of every sample, stacked; ``None`` if no sample has it.
    Raises ``ValueError`` if only some samples do."""
    values = [getattr(sample, field) for sample in batch]
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        missing = [sample.stem for sample, has in zip(batch, present, strict=True) if not has]
        raise ValueError(f"{field} is missing from some samples of the batch: {missing}")
    return torch.stack(values)
