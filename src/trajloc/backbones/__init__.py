"""Backbone adapters: everything that touches a video-model library.

Each backbone is a :class:`~trajloc.backbones.base.BackboneSpec` in
:data:`BACKBONES` (Strategy + Registry). Pipelines look a backbone up by
name and use only its spec, so they never branch on the backbone. The heavy
modules (DiffSynth, diffusers, CogVideo) are named as strings and imported on
first use, so using one backbone never imports the other's libraries.

To add a backbone ``<name>``:

1. ``trajloc/backbones/<name>/``, implementing the protocols in
   :mod:`.base`: a generator, a training entry point, a ``VideoEncoder`` and
   a ``PromptEncoder``; an ``AttentionSupervisor`` subclass, conditioning
   through a ``TrajectoryConditioner``, checkpoints through
   :mod:`.trajloc_modules`;
2. a ``BackboneSpec`` in :data:`BACKBONES` (the ``trajloc`` command then
   offers it);
3. its name in ``trajloc.datasets.layout.BackboneName``, and any extra cache
   fields in ``datasets/caches.py``, ``cached_clips.py`` and ``collate.py``;
4. ``configs/<name>.yaml``, its dependencies in ``requirements/trajloc.in``
   (and a new lock), its arm in ``scripts/preprocess/stage2/run_all.sh``,
   and its pinned code under ``external/``.

The two backbones differ in these details; each released model was trained
and evaluated with its own:

- column replacement: Wan overwrites the columns and renormalizes each row;
  CogVideoX's additive correction gives each token exactly its heatmap and
  rescales the other keys by ``1 - H``;
- the box loss: Wan's is a per-element masked mean over the batch;
  CogVideoX's is per sample and sums the 16 latent channels per cell;
- appearance injection: Wan in each sample's position order, CogVideoX in
  object order;
- the frame-0 latent: Wan's from its I2V conditioning, CogVideoX's frame 0
  encoded on its own;
- CogVideoX conditions on the ground-truth clip's first frame, falling back
  to the decoded frame-0 latent (which differs slightly) without the video;
- sharded inference numbers samples globally for Wan, per shard for CogVideoX.
"""

from __future__ import annotations

from trajloc.backbones.base import (
    BackboneSpec,
    GenerationSettings,
    Generator,
    GeneratorFactory,
    PromptEncoder,
    TokenizedPrompt,
    TrainingEntryPoint,
    VideoEncoder,
)
from trajloc.datasets.eval_samples import ShardNumbering

BACKBONES: dict[str, BackboneSpec] = {
    "wan": BackboneSpec(
        name="wan",
        display_name="Wan2.1-I2V-14B-480P",
        generator_factory="trajloc.backbones.wan.inference:WanGenerator",
        training_entry_point="trajloc.backbones.wan.training:train",
        shard_numbering=ShardNumbering.GLOBAL,
        default_guidance_scale=5.0,
        default_shift=5.0,
        checkpoint_files=("lora.safetensors", "trajloc_modules.safetensors"),
        release_repo="Omer-Sela/TrajLoc-Wan2.1-I2V-14B-480P",
        release_revision="323d93bbbab7a9f1ca237a44ccd968838cdfcef1",
        supports_resume=False,
        video_encoder="trajloc.backbones.wan.stage2:WanVideoEncoder",
        prompt_encoder="trajloc.backbones.wan.stage2:WanPromptEncoder",
    ),
    "cogvideox": BackboneSpec(
        name="cogvideox",
        display_name="CogVideoX-5B-I2V",
        generator_factory="trajloc.backbones.cogvideox.inference:CogVideoXGenerator",
        training_entry_point="trajloc.backbones.cogvideox.training:train",
        shard_numbering=ShardNumbering.PER_SHARD,
        default_guidance_scale=6.0,
        default_shift=None,
        checkpoint_files=("pytorch_lora_weights.safetensors", "trajloc_modules.safetensors"),
        release_repo="Omer-Sela/TrajLoc-CogVideoX-5B-I2V",
        release_revision="237b326bfd1f667481beb78a46f055d5d7b116c4",
        supports_resume=True,
        video_encoder="trajloc.backbones.cogvideox.stage2:CogVideoXVideoEncoder",
        prompt_encoder="trajloc.backbones.cogvideox.stage2:CogVideoXPromptEncoder",
    ),
}


def get_backbone(name: str) -> BackboneSpec:
    """The registered spec, or ``KeyError`` listing the known names."""
    if name not in BACKBONES:
        raise KeyError(f"Unknown backbone {name!r}. Known: {sorted(BACKBONES)}")
    return BACKBONES[name]


__all__ = [
    "BACKBONES",
    "BackboneSpec",
    "GenerationSettings",
    "Generator",
    "GeneratorFactory",
    "PromptEncoder",
    "TokenizedPrompt",
    "TrainingEntryPoint",
    "VideoEncoder",
    "get_backbone",
]
