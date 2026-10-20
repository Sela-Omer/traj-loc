"""The ``trajloc`` section of a training config, typed.

A training config has the backbone's own settings at the top level and
TrajLoc's under ``trajloc:``. That section is parsed once, into a
:class:`TrajLocConfig`; nothing downstream reads the raw mapping. The YAML
keys are the field names. Every key is required and unknown keys are an
error, so a config states all of its settings and a typo can't silently fall
back to a default. Values are taken as written; :func:`as_number` reads a
number a backbone's settings need, whichever way YAML spelled it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any


class BoxMaskStyle(str, Enum):
    """How object boxes become the box-masked loss's mask (``box_mask.style``)."""

    PADDED = "padded"  # each box extended in time by ``padding_frames`` (CogVideoX)
    TIGHT = "tight"  # each box at its own latent frame only (Wan)


@dataclass(frozen=True)
class BoxMaskSpec:
    """The ``box_mask`` entry: the mask's style and, for ``padded``, its extent.

    Attributes:
        style: see :class:`BoxMaskStyle`.
        padding_frames: video frames each box is extended by, before and
            after (``padded`` only; ``0`` for ``tight``).
    """

    style: BoxMaskStyle
    padding_frames: int

    @classmethod
    def from_mapping(cls, entry: Mapping[str, Any]) -> BoxMaskSpec:
        """Parse ``box_mask``: ``{style: padded, padding_frames: N}`` or ``{style: tight}``."""
        style = BoxMaskStyle(_required(entry, "style", "trajloc.box_mask"))
        if style is BoxMaskStyle.PADDED:
            _check_keys(entry, {"style", "padding_frames"}, "trajloc.box_mask")
            padding = _required(entry, "padding_frames", "trajloc.box_mask")
            if padding < 0:
                raise ValueError(f"trajloc.box_mask.padding_frames: expected >= 0, got {padding}")
            return cls(style, padding)
        _check_keys(entry, {"style"}, "trajloc.box_mask (tight takes no padding)")
        return cls(style, 0)


@dataclass(frozen=True)
class TrainingSet:
    """The ``training_set`` entry: the preprocessed dataset to train on.

    Attributes:
        name: the dataset's name (in logs and in every sample).
        root: the dataset root (Stage 1 and the backbone's Stage-2 caches;
            see :mod:`trajloc.datasets.layout`), relative to the repository.
    """

    name: str
    root: str

    @classmethod
    def from_mapping(cls, entry: Mapping[str, Any]) -> TrainingSet:
        return cls(**_parsed(entry, cls, "trajloc.training_set"))


@dataclass(frozen=True)
class TrajLocConfig:
    """TrajLoc's settings in a training config.

    Attributes:
        video_size: ``(height, width)`` of the clips, in pixels.
        num_frames: frames per clip.
        appearance_latent_size: ``(height, width)`` of the frame-0 VAE latent.
        appearance_latent_depth: the appearance encoder's latent channels.
        appearance_hidden_channels: its hidden width.
        appearance_token_std: the text encoder's token std, which the
            appearance tokens are scaled to.
        heatmap_sigma: the heatmaps' sigma (its initial value, where it is
            trained), in pixels.
        box_loss_weight: the box-masked loss's weight ``w`` in
            ``(1 - w) * mse + w * box_mse``.
        box_mask: the box mask of the box-masked loss.
        training_set: the dataset to train on.
    """

    video_size: tuple[int, int]
    num_frames: int
    appearance_latent_size: tuple[int, int]
    appearance_latent_depth: int
    appearance_hidden_channels: int
    appearance_token_std: float
    heatmap_sigma: float
    box_loss_weight: float
    box_mask: BoxMaskSpec
    training_set: TrainingSet

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any]) -> TrajLocConfig:
        """Parse a config's ``trajloc`` section."""
        values = _parsed(section, cls, "trajloc")
        values["video_size"] = tuple(values["video_size"])
        values["appearance_latent_size"] = tuple(values["appearance_latent_size"])
        values["box_mask"] = BoxMaskSpec.from_mapping(values["box_mask"])
        values["training_set"] = TrainingSet.from_mapping(values["training_set"])
        return cls(**values)

    def __post_init__(self) -> None:
        if not 0.0 <= self.box_loss_weight <= 1.0:
            raise ValueError(
                f"trajloc.box_loss_weight: expected [0, 1], got {self.box_loss_weight}"
            )


def _parsed(values: Mapping[str, Any], cls: type, context: str) -> dict[str, Any]:
    """``values`` as a dict with exactly the fields of ``cls``."""
    _check_keys(values, {f.name for f in fields(cls)}, context)
    missing = [f.name for f in fields(cls) if f.name not in values]
    if missing:
        raise ValueError(f"{context}: missing keys {missing}")
    return dict(values)


def as_number(value: Any, kind: type[int] | type[float], key: str) -> int | float:
    """``value`` as an ``int`` or ``float``, accepting a numeric string (PyYAML
    reads ``1e-4``, without a dot, as text); ``ValueError`` naming ``key`` otherwise."""
    if isinstance(value, str):
        try:
            value = float(value) if kind is float else int(value)
        except ValueError:
            raise ValueError(f"{key}: expected a number, got {value!r}") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key}: expected a number, got {value!r}")
    if kind is int:
        if value != int(value):
            raise ValueError(f"{key}: expected an integer, got {value!r}")
        return int(value)
    return float(value)


def _required(values: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in values:
        raise ValueError(f"{context}: missing key {key!r}")
    return values[key]


def _check_keys(values: Mapping[str, Any], accepted: set[str], context: str) -> None:
    unknown = [key for key in values if key not in accepted]
    if unknown:
        raise ValueError(f"{context}: unknown keys {unknown}; expected {sorted(accepted)}")


# Top-level keys specific to one run, not part of the model's settings: set by
# the training runner (``output_dir``) or for one run only. A checkpoint's
# saved config omits them.
RUN_KEYS = frozenset({"output_dir", "resume_from_checkpoint"})


def check_config_keys(config: Mapping[str, Any], required: frozenset[str], backbone: str) -> None:
    """Raise ``ValueError`` unless ``config`` states every key of ``required``
    and has no other key besides :data:`RUN_KEYS`."""
    unknown = sorted(key for key in config if key not in required | RUN_KEYS)
    missing = sorted(key for key in required if key not in config)
    if unknown or missing:
        raise ValueError(f"{backbone} config: unknown keys {unknown}, missing keys {missing}")
