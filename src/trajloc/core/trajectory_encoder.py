"""The frozen trajectory encoder: per-object trajectory -> one text token.

Each object's ``<traj_i>`` token is replaced, at Stage 2, by an embedding of
its whole trajectory. The encoder was pretrained separately (on synthetic
trajectories, through a T5 reconstruction objective) and is frozen here. It
ships as ``checkpoints/trajectory_encoder.safetensors``.

Architecture (must match the checkpoint): three stride-2 1D convolutions over
time (64, 128, 256 channels) on ``(x, y, depth, orientation, t)``, flattened,
then linear ``mu`` / ``logvar`` heads (32-D). A two-layer MLP projects ``mu`` to
the 4096-D text width. The encoder expects 89 frames. The orientation channel
exists in the checkpoint but is fed zeros, which the encoder saw half the time
during pretraining.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from trajloc.core.token_scale import rescale_to_std


def _conv_temporal_out(num_frames: int, num_stride2_layers: int) -> int:
    t = num_frames
    for _ in range(num_stride2_layers):
        t = math.ceil(t / 2)
    return t


NUM_TRAJECTORY_TOKENS = 200  # <traj_0> .. <traj_199> are added to both text tokenizers


def trajectory_token_strings(n: int = NUM_TRAJECTORY_TOKENS) -> list[str]:
    """The ``<traj_i>`` token strings added to the text tokenizers."""
    return [f"<traj_{i}>" for i in range(n)]


class TrajectoryEncoder(nn.Module):
    """``(N, T, coords_dim)`` trajectories -> ``(mu, logvar)``, each ``(N, latent_dim)``.

    A time channel ``linspace(0, 1, T)`` is appended inside ``forward``;
    callers pass coordinates only.

    Args:
        num_frames: T (89 for the released checkpoint).
        coords_dim: input channels (4: x, y, depth, orientation).
        hidden: width of the first conv (then 2x, 4x).
        latent_dim: size of ``mu`` / ``logvar``.
    """

    def __init__(
        self,
        num_frames: int = 89,
        coords_dim: int = 4,
        hidden: int = 64,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.coords_dim = coords_dim
        self.hidden = hidden
        self.latent_dim = latent_dim

        in_ch = coords_dim + 1
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden * 2, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.GELU(),
            nn.Conv1d(hidden * 2, hidden * 4, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden * 4),
            nn.GELU(),
        )
        t_out = _conv_temporal_out(num_frames, 3)
        fc_in = hidden * 4 * t_out
        self.fc_mu = nn.Linear(fc_in, latent_dim)
        self.fc_logvar = nn.Linear(fc_in, latent_dim)

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``coords``: (N, T, coords_dim) → (mu, logvar) each (N, latent_dim)."""
        N, T, _ = coords.shape
        t_pos = torch.linspace(0, 1, T, device=coords.device).view(1, T, 1).expand(N, -1, -1)
        x = torch.cat([coords, t_pos], dim=-1).permute(0, 2, 1)
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class TrajectoryProjection(nn.Module):
    """Two-layer MLP from the trajectory latent to the text-embedding width.

    The last layer starts at ``N(0, 0.02^2)`` weights and zero bias (only
    relevant when training it; the released one is loaded frozen).
    """

    def __init__(self, latent_dim: int = 32, text_embed_dim: int = 4096) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, text_embed_dim),
        )
        nn.init.normal_(self.layers[-1].weight, std=0.02)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(..., latent_dim)`` trajectory latents -> ``(..., text_embed_dim)``."""
        tokens: torch.Tensor = self.layers(z)
        return tokens


class TrajectoryTokenEncoder(nn.Module):
    """Trajectories -> text-space trajectory tokens, as Stage 2 caches them.

    ``forward(coords (B, N, T, 2), depths (B, N, T)) -> (B, N, 4096)``, with
    coordinates and depths in [0, 1] and -1 marking invalid frames:

    1. resample time from T to the encoder's 89 frames (linear, corners aligned);
    2. build ``(x, y, depth, 0)``, clamping missing depths to 0, and zero every
       invalid frame (x < 0 OR y < 0, after resampling; note the heatmap
       generator's rule differs);
    3. encode, take ``mu`` (no sampling), project;
    4. normalize over the whole output to ``token_std`` (the target
       text encoder's token std: 0.15 CogVideoX, 0.0734 Wan).
    """

    def __init__(
        self,
        encoder: TrajectoryEncoder,
        projection: TrajectoryProjection,
        *,
        token_std: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.token_std = token_std
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        coords: torch.Tensor,  # (B, N, T, 2)
        depths: torch.Tensor,  # (B, N, T)
    ) -> torch.Tensor:  # (B, N, text_embed_dim)
        """Per object, the embedding that replaces its ``<traj_i>`` token,
        ``(B, N, text_embed_dim)``."""
        if coords.dim() != 4 or coords.shape[-1] != 2:
            raise ValueError(f"coords must be (B, N, T, 2); got {tuple(coords.shape)}")
        if depths.dim() != 3 or depths.shape[:3] != coords.shape[:3]:
            raise ValueError(
                f"depths must be (B, N, T) matching coords; got "
                f"{tuple(depths.shape)} vs {tuple(coords.shape)}"
            )
        B, N, T, _ = coords.shape
        target_T = self.encoder.num_frames

        if T != target_T:
            c = coords.reshape(B * N, T, 2).permute(0, 2, 1)
            c = torch.nn.functional.interpolate(
                c,
                size=target_T,
                mode="linear",
                align_corners=True,
            )
            coords_resampled = c.permute(0, 2, 1).reshape(B, N, target_T, 2)

            d = depths.reshape(B * N, 1, T)
            d = torch.nn.functional.interpolate(
                d,
                size=target_T,
                mode="linear",
                align_corners=True,
            )
            depths_resampled = d.reshape(B, N, target_T)
        else:
            coords_resampled = coords
            depths_resampled = depths

        valid = (coords_resampled[..., 0] >= 0) & (coords_resampled[..., 1] >= 0)  # (B, N, T')
        depths_clamped = depths_resampled.clone()
        depths_clamped = torch.where(
            depths_clamped < 0, torch.zeros_like(depths_clamped), depths_clamped
        )
        zeros = torch.zeros_like(depths_clamped)
        channels = torch.stack(
            [coords_resampled[..., 0], coords_resampled[..., 1], depths_clamped, zeros], dim=-1
        )  # (B, N, T', 4)
        channels = channels * valid.float().unsqueeze(-1)
        flat = channels.reshape(B * N, target_T, 4)

        mu, _logvar = self.encoder(flat)  # (B*N, latent_dim)
        embeddings = self.projection(mu)  # (B*N, text_embed_dim)

        # One mean and std over the whole (B*N, 4096) output.
        embeddings = rescale_to_std(embeddings, self.token_std)

        return embeddings.view(B, N, -1)


def load_trajectory_token_encoder(
    checkpoint_path: str | Path,
    *,
    token_std: float,
    num_frames: int = 89,
    coords_dim: int = 4,
    hidden: int = 64,
    latent_dim: int = 32,
    text_embed_dim: int = 4096,
    device: str = "cpu",
) -> TrajectoryTokenEncoder:
    """The frozen trajectory encoder of a checkpoint, on ``device``, in eval mode.

    The checkpoint is a :class:`TrajectoryTokenEncoder`'s state dict as
    safetensors (``encoder.*``, ``projection.*``); the architecture
    arguments must match it.
    """
    from safetensors.torch import load_file

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"no trajectory encoder at {checkpoint_path}: it ships with each released model "
            f"(scripts/prepare_data.sh links or downloads it into checkpoints/)"
        )
    encoder = TrajectoryEncoder(
        num_frames=num_frames, coords_dim=coords_dim, hidden=hidden, latent_dim=latent_dim
    )
    projection = TrajectoryProjection(latent_dim=latent_dim, text_embed_dim=text_embed_dim)
    token_encoder = TrajectoryTokenEncoder(encoder, projection, token_std=token_std)
    token_encoder.load_state_dict(load_file(str(checkpoint_path), device=device))
    return token_encoder.to(device).eval()
