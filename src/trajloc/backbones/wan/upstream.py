"""The upstream Wan code the training module is built on: the one module
that imports DiffSynth-Studio.

Both checkouts, Wan2.1 and DiffSynth-Studio, are used unmodified from
``external/`` (git submodules at pinned commits). Importing this module prepares
them (:func:`prepare`) and imports the DiffSynth names TrajLoc uses,
re-exported below; ``wan`` (Wan2.1) is importable afterwards. Only the Wan
model and inference code import this module, on first use.

One deliberate exception: the supervised attention
(:mod:`~trajloc.backbones.wan.attention`) imports DiffSynth's flash
attention at call time, so the supervisor can be imported without
DiffSynth.
"""

from __future__ import annotations

from trajloc.paths import add_to_path

_prepared = False


def prepare() -> None:
    """Put ``external/Wan2.1`` and then ``external/DiffSynth-Studio`` first on
    ``sys.path`` (so DiffSynth ends up first). Runs once per process."""
    global _prepared
    if _prepared:
        return
    add_to_path("external/Wan2.1", "external/DiffSynth-Studio")
    _prepared = True


prepare()

from diffsynth.diffusion.training_module import DiffusionTrainingModule  # noqa: E402
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402

__all__ = ["DiffusionTrainingModule", "ModelConfig", "WanVideoPipeline", "prepare"]
