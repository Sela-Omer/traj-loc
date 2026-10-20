"""The upstream CogVideo finetuning code: the one module that imports it.

The pinned CogVideo checkout (``external/CogVideo``) is used unmodified.
Importing this module prepares it (:func:`prepare`: ``sys.path``, and a no-op
``wandb`` stub, since ``finetune/trainer.py`` imports ``wandb``
unconditionally) and imports the names TrajLoc uses, re-exported below. Those
side effects are why nothing else imports ``finetune`` directly, and why this
module is only imported by the CogVideoX training code, on first use.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

from trajloc.paths import add_to_path


def install_wandb_stub_if_missing() -> bool:
    """Put a no-op ``wandb`` module in ``sys.modules`` unless the real one
    is importable. Returns whether the stub was installed. Idempotent.

    The stub has a real ``ModuleSpec`` (diffusers calls ``find_spec``) and
    ``Image`` / ``Video`` classes, which the upstream trainer only uses in
    ``isinstance`` checks.
    """
    try:
        import wandb  # noqa: F401
    except ImportError:
        pass
    else:
        return False

    if "wandb" in sys.modules:
        return False

    stub = types.ModuleType("wandb")
    # diffusers calls importlib.util.find_spec("wandb"), which raises if
    # __spec__ is None. Give the stub a real ModuleSpec.
    stub.__spec__ = importlib.machinery.ModuleSpec(name="wandb", loader=None)
    stub.__path__ = []
    vars(stub)["__version__"] = "0.0.0+stub"

    class _StubArtifact:
        """Stand-in for ``wandb.Image`` / ``wandb.Video``. The upstream
        code uses these in ``isinstance(...)`` checks for filtering, so
        the only requirement is that the class exists.
        """

        def __init__(self, *_args, **_kwargs) -> None:
            pass

    stub.Image = _StubArtifact  # type: ignore[attr-defined]
    stub.Video = _StubArtifact  # type: ignore[attr-defined]
    stub.init = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    stub.log = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    stub.finish = lambda *args, **kwargs: None  # type: ignore[attr-defined]

    sys.modules["wandb"] = stub
    return True


_prepared = False


def prepare() -> None:
    """Put ``external/CogVideo`` and ``external/CogVideo/finetune`` on
    ``sys.path`` (in that order, each inserted first) and install the wandb
    stub if needed. Runs once per process."""
    global _prepared
    if _prepared:
        return
    add_to_path("external/CogVideo", "external/CogVideo/finetune")
    install_wandb_stub_if_missing()
    _prepared = True


prepare()

from finetune.models.cogvideox_i2v.lora_trainer import CogVideoXI2VLoraTrainer  # noqa: E402
from finetune.schemas import Args  # noqa: E402
from finetune.utils import unwrap_model  # noqa: E402
from finetune.utils.optimizer_utils import get_optimizer  # noqa: E402

__all__ = [
    "Args",
    "CogVideoXI2VLoraTrainer",
    "get_optimizer",
    "install_wandb_stub_if_missing",
    "prepare",
    "unwrap_model",
]
