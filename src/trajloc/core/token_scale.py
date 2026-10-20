"""Matching a token's scale to a text encoder's."""

from __future__ import annotations

import torch


def rescale_to_std(tokens: torch.Tensor, target_std: float) -> torch.Tensor:
    """``tokens`` standardized over all their elements and rescaled to
    ``target_std`` (the standard deviation floored at 1e-6)."""
    mean = tokens.mean()
    std = tokens.std().clamp(min=1e-6)
    return (tokens - mean) / std * target_std
