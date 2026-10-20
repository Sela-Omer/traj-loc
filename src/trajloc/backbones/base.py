"""The contract between the pipelines and a backbone (Strategy).

A backbone is a :class:`BackboneSpec`: how to build its :class:`Generator`
for inference, its training entry point, its Stage-2 encoders
(:class:`VideoEncoder`, :class:`PromptEncoder`), and the few settings in
which the backbones legitimately differ. The pipelines depend on this module
only, never on a backbone's own modules, which are imported lazily by name.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from trajloc.datasets.eval_samples import EvalSample, ShardNumbering
from trajloc.datasets.layout import BackboneName

if TYPE_CHECKING:  # torch only for the annotations: the CLIs import this module
    import torch

    from trajloc.datasets.caches import LatentCache
    from trajloc.datasets.layout import ClipFiles


@dataclass(frozen=True)
class GenerationSettings:
    """What a generator needs to know about the run.

    Args:
        checkpoint_dir: a ``checkpoint-<step>/`` directory.
        device: torch device string.
        guidance_scale: classifier-free guidance scale.
        num_steps: sampling steps.
        shift: flow-matching scheduler shift (backbones without one ignore it).
    """

    checkpoint_dir: str | Path
    device: str
    guidance_scale: float
    num_steps: int
    shift: float | None


# A generated video: ``(T, H, W, 3)`` uint8 RGB frames.
VideoFrames = np.ndarray[Any, np.dtype[np.uint8]]


class Generator(Protocol):
    """A loaded model that turns evaluation samples into videos."""

    def generate(self, sample: EvalSample, seed: int) -> VideoFrames:
        """``(T, H, W, 3)`` uint8 RGB frames for one sample."""


class GeneratorFactory(Protocol):
    """A callable that loads a backbone's :class:`Generator`."""

    def __call__(self, config: dict[str, Any], settings: GenerationSettings) -> Generator:
        """Load the model described by the training ``config`` + checkpoint."""


class TrainingEntryPoint(Protocol):
    """A backbone's training entry point."""

    def __call__(self, config: dict[str, Any], run_dir: Path) -> None:
        """Train per ``config``. Checkpoints go to ``config["output_dir"]``
        (``run_dir/checkpoints``); ``run_dir/logs`` is there for the
        backbone's own logs (Wan's loss trace)."""


class VideoEncoder(Protocol):
    """A backbone's VAE, for Stage 2's latent caches."""

    def load(self, device: str) -> None:
        """Load the VAE on ``device``."""

    def encode(self, video: torch.Tensor, source_video: str) -> LatentCache:
        """The latent cache of a ``(T, 3, H, W)`` clip in [-1, 1]."""


@dataclass(frozen=True)
class TokenizedPrompt:
    """A tokenized prompt.

    Attributes:
        input_ids: ``(1, L)`` ids on the device, what the encoder reads.
        token_ids: the ids as a list (for the ``<traj_i>`` check).
        offsets: per token, its ``(start, end)`` character span.
        attention_mask: ``(1, L)`` on the device, if the encoder takes one.
        seq_len: the unpadded length, if the cache keeps only that.
    """

    input_ids: torch.Tensor
    token_ids: list[int]
    offsets: list[tuple[int, int]]
    attention_mask: torch.Tensor | None = None
    seq_len: int | None = None


class PromptEncoder(Protocol):
    """A backbone's tokenizer and text encoder, for Stage 2's text caches.

    ``text_token_std`` is the text encoder's token std; the trajectory
    embeddings written into the prompt are scaled to it.
    """

    text_token_std: float

    def load(self, device: str) -> None:
        """Load the tokenizer and text encoder, with the ``<traj_i>`` tokens
        added, then the models :meth:`extra_fields` needs."""

    def tokenize(self, prompt: str, stem: str) -> TokenizedPrompt:
        """Tokenize ``prompt`` the way the encoder will read it."""

    def trajectory_token_id(self, index: int) -> int:
        """The id of ``<traj_{index}>``."""

    def input_embeddings(self, tokenized: TokenizedPrompt) -> torch.Tensor:
        """``(1, L, D)`` input embeddings of the tokens."""

    def encode(self, input_embeds: torch.Tensor, tokenized: TokenizedPrompt) -> torch.Tensor:
        """The prompt embedding cached for the clip, ``(L', D)`` float32 on CPU."""

    def extra_fields(self, files: ClipFiles) -> dict[str, Any]:
        """The :class:`~trajloc.datasets.caches.TextCache` fields only this
        backbone caches."""

    def report(self, stem: str, payload: dict[str, Any]) -> None:
        """Log a human-readable trace of one clip's cache."""


@dataclass(frozen=True)
class BackboneSpec:
    """How the pipelines drive one backbone.

    Attributes:
        name: the backbone's id (``runs/<stage>/<name>/``, and its caches'
            ``cache/<name>/``: :data:`trajloc.datasets.layout.BackboneName`).
        display_name: its name in prose (help texts).
        generator_factory: ``"module:attr"`` of a :class:`GeneratorFactory`.
        training_entry_point: ``"module:attr"`` of a :class:`TrainingEntryPoint`.
        shard_numbering: how sharded inference numbers its samples.
        default_guidance_scale: the released inference setting.
        default_shift: the released scheduler shift; ``None`` if the
            backbone has no shift.
        checkpoint_files: the files a checkpoint directory must hold for
            inference (checked before the model is loaded).
        release_repo: the released model's Hugging Face Hub repository, the
            commands' default checkpoint (:mod:`trajloc.hub`).
        release_revision: the commit of it the code was validated with.
        supports_resume: whether training can continue from a checkpoint
            (the config's ``resume_from_checkpoint``).
        video_encoder: ``"module:attr"`` of its Stage-2 :class:`VideoEncoder` class.
        prompt_encoder: ``"module:attr"`` of its Stage-2 :class:`PromptEncoder` class.
    """

    name: BackboneName
    display_name: str
    generator_factory: str
    training_entry_point: str
    shard_numbering: ShardNumbering
    default_guidance_scale: float
    default_shift: float | None
    checkpoint_files: tuple[str, ...]
    release_repo: str
    release_revision: str
    supports_resume: bool
    video_encoder: str
    prompt_encoder: str

    def load_generator(self, config: dict[str, Any], settings: GenerationSettings) -> Generator:
        """Import the backbone's generator and load the model."""
        return cast(Generator, import_object(self.generator_factory)(config, settings))

    def train(self, config: dict[str, Any], run_dir: Path) -> None:
        """Import the backbone's training entry point and run it."""
        import_object(self.training_entry_point)(config, run_dir)

    def load_video_encoder(self) -> VideoEncoder:
        """A new (not yet loaded) Stage-2 :class:`VideoEncoder`."""
        return cast(VideoEncoder, import_object(self.video_encoder)())

    def load_prompt_encoder(self) -> PromptEncoder:
        """A new (not yet loaded) Stage-2 :class:`PromptEncoder`."""
        return cast(PromptEncoder, import_object(self.prompt_encoder)())

    def missing_checkpoint_files(self, checkpoint_dir: Path) -> list[str]:
        """The :attr:`checkpoint_files` that ``checkpoint_dir`` lacks."""
        return [f for f in self.checkpoint_files if not (Path(checkpoint_dir) / f).is_file()]


def import_object(target: str) -> Any:
    """The object a lazy ``"package.module:attr"`` reference names (the
    registries use these so that listing them imports no backbone)."""
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)
