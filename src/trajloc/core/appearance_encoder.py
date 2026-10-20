"""Appearance encoder: one text-space token per object, from the first frame.

Each object's text token (its noun, e.g. "dog") is overwritten with an
embedding of what the object looks like in frame 0, so the model can tell
objects of the same class apart. From the frame-0 VAE latent:

1. a small 2D conv encoder (stride 2) with VAE-style ``mu`` / ``logvar`` heads
   (sampled in training, ``mu`` at eval) gives a compressed feature map;
2. the feature vector at each object's first visible position is picked;
3. ``to_text`` projects it to the text-embedding width (4096);
4. the projected tokens are renormalized to the text encoder's token std
   (0.15 for CogVideoX's T5-XXL, 0.0734 for Wan's UMT5-XXL), over the real
   (unpadded) tokens of each sample.

The final projection starts at ``N(0, 0.02^2)`` weights and zero bias, so the
appearance signal is near zero at initialization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from trajloc.core.config import TrajLocConfig
from trajloc.core.token_scale import rescale_to_std


class AppearanceEncoder(nn.Module):
    """Frame-0 VAE latent + object start positions -> one text token per object.

    Args:
        vae_channels: VAE latent channels (16 for both backbones).
        latent_height, latent_width: VAE latent size (60 x 90 for 480 x 720 video).
        latent_depth: bottleneck channels of the ``mu`` / ``logvar`` heads.
        hidden_channels: conv encoder width.
        text_embed_dim: text-embedding width (4096 for both backbones).
        token_std: the std the projected tokens are rescaled to (the text
            encoder's: 0.15 CogVideoX, 0.0734 Wan).
    """

    def __init__(
        self,
        vae_channels: int = 16,
        latent_height: int = 60,
        latent_width: int = 90,
        latent_depth: int = 8,
        hidden_channels: int = 64,
        text_embed_dim: int = 4096,
        *,
        token_std: float,
    ) -> None:
        super().__init__()
        self.vae_channels = vae_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.latent_depth = latent_depth
        self.text_embed_dim = text_embed_dim
        self.token_std = token_std

        width = hidden_channels

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(vae_channels, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
        )
        self.compressed_height = math.ceil(latent_height / 2)
        self.compressed_width = math.ceil(latent_width / 2)

        self.conv_mu = nn.Conv2d(width, latent_depth, 1)
        self.conv_logvar = nn.Conv2d(width, latent_depth, 1)

        self.to_text = nn.Linear(latent_depth, text_embed_dim)
        nn.init.normal_(self.to_text.weight, std=0.02)
        nn.init.zeros_(self.to_text.bias)

    @classmethod
    def from_config(cls, config: TrajLocConfig) -> AppearanceEncoder:
        """The appearance encoder of a TrajLoc config."""
        latent_height, latent_width = config.appearance_latent_size
        return cls(
            latent_height=latent_height,
            latent_width=latent_width,
            latent_depth=config.appearance_latent_depth,
            hidden_channels=config.appearance_hidden_channels,
            token_std=config.appearance_token_std,
        )

    def encode_frame(self, frame0_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, C, H, W)`` latent -> ``(mu, logvar)``, each ``(B, depth, H/2, W/2)``."""
        _, C, H, W = frame0_latent.shape
        if H != self.latent_height or W != self.latent_width:
            raise ValueError(
                f"Frame-0 latent shape ({H}, {W}) doesn't match config "
                f"({self.latent_height}, {self.latent_width})"
            )
        if C != self.vae_channels:
            raise ValueError(
                f"Frame-0 latent channels {C} doesn't match config {self.vae_channels}"
            )
        h = self.encoder_conv(frame0_latent.to(dtype=next(self.encoder_conv.parameters()).dtype))
        return self.conv_mu(h), self.conv_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """A sample ``mu + eps * exp(logvar / 2)`` in training, ``mu`` otherwise."""
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def sample_positions(self, z: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Feature vectors at integer ``(i, j)`` positions: ``(B, N, depth)``."""
        B, _, H, W = z.shape
        N = positions.shape[1]
        batch_idx = torch.arange(B, device=z.device).view(B, 1).expand(-1, N)
        i_idx = positions[:, :, 0].clamp(0, H - 1)
        j_idx = positions[:, :, 1].clamp(0, W - 1)
        return z[batch_idx, :, i_idx, j_idx]

    def project_to_text(
        self,
        sampled: torch.Tensor,
        num_objects: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project to text width and renormalize. For ``B > 1``, sample ``b``
        is normalized over its first ``num_objects[b]`` tokens only, so
        padding doesn't leak into the statistics."""
        if num_objects is not None and (num_objects == 0).any():
            raise ValueError(
                f"num_objects contains zeros: {num_objects.tolist()} — filter before batching"
            )

        emb = self.to_text(sampled)

        B = emb.shape[0]
        if B == 1:
            emb = self._normalize(emb)
        else:
            normalized_parts = []
            for b in range(B):
                n_real = num_objects[b].item() if num_objects is not None else emb.shape[1]
                emb_b = emb[b]
                real = self._normalize(emb_b[:n_real])
                emb_b_out = emb_b.clone()
                emb_b_out[:n_real] = real
                normalized_parts.append(emb_b_out)
            emb = torch.stack(normalized_parts)

        if torch.isnan(emb).any():
            raise RuntimeError("NaN in appearance encoder output after normalization")
        return emb

    def _normalize(self, t: torch.Tensor) -> torch.Tensor:
        return rescale_to_std(t, self.token_std)

    def forward(
        self,
        frame0_latent: torch.Tensor,
        trajectory_starts: torch.Tensor,
        num_objects: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One text-space token per object, ``(B, N, text_embed_dim)``: the
        feature at each object's start on the compressed frame-0 map, projected
        and scaled to ``token_std``.

        Args:
            frame0_latent: ``(B, 16, 60, 90)``.
            trajectory_starts: ``(B, N, 2)`` first visible ``(x, y)`` in [0, 1].
            num_objects: ``(B,)`` real object counts (for padded batches).
        """
        mu, logvar = self.encode_frame(frame0_latent)
        z = self.reparameterize(mu, logvar)

        pos_j = (
            (trajectory_starts[:, :, 0] * (self.compressed_width - 1))
            .round()
            .long()
            .clamp(0, self.compressed_width - 1)
        )
        pos_i = (
            (trajectory_starts[:, :, 1] * (self.compressed_height - 1))
            .round()
            .long()
            .clamp(0, self.compressed_height - 1)
        )
        positions = torch.stack([pos_i, pos_j], dim=-1)

        sampled = self.sample_positions(z, positions)
        emb = self.project_to_text(sampled, num_objects=num_objects)

        return emb


# Writing the appearance tokens into a prompt embedding: an
# :data:`~trajloc.core.conditioning.AppearanceInjection`. The backbones differ
# in how they walk the positions; the two rules agree whenever every object's
# position is distinct and inside the prompt.


def inject_in_position_order(
    prompt_embeddings: torch.Tensor,
    appearance_tokens: torch.Tensor,
    positions: list[dict[int, int]],
    num_objects: torch.Tensor,
) -> torch.Tensor:
    """Wan's rule. For each sample ``b``, in the order of ``positions[b]``
    (object index -> text position), write ``appearance_tokens[b, i]`` at the
    position; entries outside either tensor are skipped (``num_objects``
    is not used). Returns a copy."""
    prompt_embeddings = prompt_embeddings.clone()
    for b in range(len(positions)):
        for object_index, pos in positions[b].items():
            ti = int(object_index)
            p = int(pos)
            if ti < appearance_tokens.shape[1] and p < prompt_embeddings.shape[1]:
                prompt_embeddings[b, p, :] = appearance_tokens[b, ti, :].to(prompt_embeddings.dtype)
    return prompt_embeddings


def inject_in_object_order(
    prompt_embeddings: torch.Tensor,
    appearance_tokens: torch.Tensor,
    positions: list[dict[int, int]],
    num_objects: torch.Tensor,
) -> torch.Tensor:
    """CogVideoX's rule. For each sample ``b`` and object ``i = 0 .. num_objects[b] - 1``
    that has an entry in ``positions[b]``, write ``appearance_tokens[b, i]`` at its
    position, in index order. Returns a copy."""
    counts = [int(num_objects[b].item()) for b in range(len(positions))]
    prompt_embeddings = prompt_embeddings.clone()
    for b, n in enumerate(counts):
        for i in range(n):
            if i in positions[b]:
                pos = positions[b][i]
                prompt_embeddings[b, pos, :] = appearance_tokens[b, i, :].to(
                    prompt_embeddings.dtype
                )
    return prompt_embeddings
