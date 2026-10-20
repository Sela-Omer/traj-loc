"""The Stage-2 cache files: the one definition of their contents.

::

    <root>/cache/<backbone>/
      text/<stem>.pt      TextCache: prompt embedding, token positions, trajectories
      latents/<stem>.pt   LatentCache: VAE latents

Each file is a ``torch.save``'d dict: a record below, its fields in order
(:meth:`to_payload`). :meth:`load` reads a file back, rejecting missing and
unknown keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import torch

from trajloc.datasets.layout import CLIP_HEIGHT, CLIP_NUM_FRAMES, CLIP_WIDTH


@dataclass(frozen=True)
class TextCache:
    """One clip's text-encoder cache.

    Attributes:
        prompt_embedding: ``(L, 4096)`` float32, the text encoder's output
            with each ``<traj_i>`` input embedding replaced by object ``i``'s
            trajectory embedding (CogVideoX: padded to 226 tokens; Wan:
            unpadded).
        trajectory_token_positions: object ``i`` -> position of its ``<traj_i>`` token.
        noun_token_positions: object ``i`` -> position of the first token of
            its head noun.
        trajectory_coords: ``(N, 49, 2)`` in [0, 1]; -1 where not visible.
        trajectory_depths: ``(N, 49)``; -1 where unknown.
        trajectory_starts: ``(N, 2)``, each object's first visible position.
        num_objects: ``N``.
        prompt: the prompt, with its ``<traj_i>`` tokens.
        stem: the clip.
        dataset_name: the dataset's name.
        object_labels: per object, its label.
        object_ids: per object, its key in ``trajectories/<stem>.json``.
        i2v_conditioning: Wan only: ``(4 + 16, 13, 60, 90)``, the I2V input.
        image_embeddings: Wan only: frame 0's CLIP image tokens.
    """

    prompt_embedding: torch.Tensor
    trajectory_token_positions: dict[int, int]
    noun_token_positions: dict[int, int]
    trajectory_coords: torch.Tensor
    trajectory_depths: torch.Tensor
    trajectory_starts: torch.Tensor
    num_objects: int
    prompt: str
    stem: str
    dataset_name: str
    object_labels: list[str]
    object_ids: list[str]
    i2v_conditioning: torch.Tensor | None = None
    image_embeddings: torch.Tensor | None = None

    _OPTIONAL: ClassVar[frozenset[str]] = frozenset({"i2v_conditioning", "image_embeddings"})

    def to_payload(self) -> dict[str, Any]:
        """The dict written to disk (the Wan-only entries only when set)."""
        return _payload(self, self._OPTIONAL)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TextCache:
        """The record of an on-disk dict (``KeyError`` on missing or unknown keys)."""
        return _from_payload(cls, payload)

    @classmethod
    def load(cls, path: str | Path) -> TextCache:
        """Read ``path``."""
        return cls.from_payload(torch.load(path, map_location="cpu", weights_only=True))


@dataclass(frozen=True)
class LatentCache:
    """One clip's VAE latents.

    Attributes:
        latents: ``(16, 13, 60, 90)`` float32.
        frame0_latent: ``(16, 60, 90)``. CogVideoX's appearance encoder reads
            it (frame 0 encoded on its own). For Wan it is ``latents[:, 0]``
            and unused: Wan's appearance encoder reads the frame-0 latent
            inside its I2V conditioning.
        num_frames, height, width: the latent grid.
        source_video: the encoded file's name.
        source_frames, source_height, source_width: its geometry.
        vae: which VAE encoded it.
        scaling_factor: the factor the latents were multiplied by (``None``: none).
        frame0_encoding: CogVideoX only: how ``frame0_latent`` was encoded.
    """

    latents: torch.Tensor
    frame0_latent: torch.Tensor
    num_frames: int
    height: int
    width: int
    source_video: str
    source_frames: int
    source_height: int
    source_width: int
    vae: str
    scaling_factor: float | None
    frame0_encoding: str | None = None

    _OPTIONAL: ClassVar[frozenset[str]] = frozenset({"frame0_encoding"})

    @classmethod
    def of_clip(
        cls,
        latents: torch.Tensor,
        frame0_latent: torch.Tensor,
        *,
        source_video: str,
        vae: str,
        scaling_factor: float | None,
        frame0_encoding: str | None = None,
    ) -> LatentCache:
        """The record of a Stage-1 clip's ``(C, F, H, W)`` latents (the clip
        geometry is every clip's: :data:`~trajloc.datasets.layout.CLIP_NUM_FRAMES`
        frames of ``CLIP_HEIGHT x CLIP_WIDTH``)."""
        _, num_frames, height, width = latents.shape
        return cls(
            latents=latents,
            frame0_latent=frame0_latent,
            num_frames=num_frames,
            height=height,
            width=width,
            source_video=source_video,
            source_frames=CLIP_NUM_FRAMES,
            source_height=CLIP_HEIGHT,
            source_width=CLIP_WIDTH,
            vae=vae,
            scaling_factor=scaling_factor,
            frame0_encoding=frame0_encoding,
        )

    def to_payload(self) -> dict[str, Any]:
        """The dict written to disk (``frame0_encoding`` only when set)."""
        return _payload(self, self._OPTIONAL)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LatentCache:
        """The record of an on-disk dict (``KeyError`` on missing or unknown keys)."""
        return _from_payload(cls, payload)

    @classmethod
    def load(cls, path: str | Path) -> LatentCache:
        """Read ``path``."""
        return cls.from_payload(torch.load(path, map_location="cpu", weights_only=True))


_Record = TypeVar("_Record", "TextCache", "LatentCache")


def _payload(record: Any, optional: frozenset[str]) -> dict[str, Any]:
    return {
        f.name: getattr(record, f.name)
        for f in fields(record)
        if not (f.name in optional and getattr(record, f.name) is None)
    }


def _from_payload(cls: type[_Record], payload: Mapping[str, Any]) -> _Record:
    by_key = {f.name: f for f in fields(cls)}
    required = {key for key, f in by_key.items() if f.default is MISSING}
    missing, unknown = required - set(payload), set(payload) - set(by_key)
    if missing or unknown:
        raise KeyError(f"{cls.__name__}: missing keys {sorted(missing)}, unknown {sorted(unknown)}")
    return cls(**{key: payload[key] for key in by_key if key in payload})
