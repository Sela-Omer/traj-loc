"""Wan checkpoints: a directory with these files (a released model adds the
trajectory encoder, ``trajectory_encoder.safetensors``).

- ``lora.safetensors``: the trainable DiT parameters (LoRA), keys without
  the ``pipe.dit.`` prefix.
- ``trajloc_modules.safetensors`` and ``config.yaml``: TrajLoc's modules and
  the training config (:mod:`trajloc.backbones.trajloc_modules`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from safetensors.torch import load_file, save_file

from trajloc.backbones.trajloc_modules import (
    load_trajloc_modules,
    save_checkpoint_config,
    save_trajloc_modules,
)

if TYPE_CHECKING:  # the model module imports DiffSynth
    from trajloc.backbones.wan.model import WanTrajLocModule

logger = logging.getLogger(__name__)

LORA_FILE = "lora.safetensors"
# The key prefixes tried when loading: this module's (``save``), and the same
# keys under PEFT's model wrapper (``base_model.model.``), as a PEFT-wrapped
# DiT's state dict names them.
LORA_KEY_PREFIXES = ("pipe.dit.", "pipe.dit.base_model.model.")


class WanCheckpointer:
    """Writes and reads Wan checkpoints.

    Args:
        save_interval: :meth:`should_save` is true at its positive multiples.
        config: the training config, saved with every checkpoint.
        remove_prefix: stripped from the exported LoRA keys.
    """

    def __init__(
        self, save_interval: int, config: dict[str, Any], remove_prefix: str = "pipe.dit."
    ) -> None:
        self.save_interval = save_interval
        self.config = config
        self.remove_prefix = remove_prefix

    def should_save(self, step: int) -> bool:
        """Whether ``step`` is a positive multiple of the interval."""
        return step > 0 and step % self.save_interval == 0

    def save(self, model: WanTrajLocModule, step: int, output_dir: str | Path) -> str:
        """Write ``output_dir/checkpoint-<step>/``; returns its path."""
        checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # The DiT's trainable parameters only (the LoRA); TrajLoc's modules
        # have their own file.
        state = model.state_dict()
        dit_state = {k: v for k, v in state.items() if k.startswith(self.remove_prefix)}
        lora = model.export_trainable_state_dict(dit_state, self.remove_prefix)
        save_file(lora, str(checkpoint_dir / LORA_FILE))
        save_trajloc_modules(checkpoint_dir, model.supervisor, model.appearance_encoder)
        save_checkpoint_config(checkpoint_dir, self.config)
        logger.info(f"Saved Wan checkpoint: {checkpoint_dir}")
        return str(checkpoint_dir)

    @staticmethod
    def load(model: WanTrajLocModule, checkpoint_dir: str | Path) -> None:
        """Load a checkpoint into a
        :class:`~trajloc.backbones.wan.model.WanTrajLocModule`.

        LoRA keys are matched as saved, then with each of
        ``LORA_KEY_PREFIXES``. A missing file, or a saved key that matches no
        parameter, raises: the model must not run with part of its weights
        freshly initialized.
        """
        checkpoint_dir = Path(checkpoint_dir)
        load_trajloc_modules(checkpoint_dir, model.supervisor, model.appearance_encoder)

        lora_path = checkpoint_dir / LORA_FILE
        if not lora_path.exists():
            raise FileNotFoundError(f"No LoRA weights at {lora_path}")
        saved = load_file(str(lora_path))
        model_state = model.state_dict()
        loaded = 0
        for key, value in saved.items():
            if key in model_state:
                model_state[key] = value
                loaded += 1
                continue
            for prefix in LORA_KEY_PREFIXES:
                if prefix + key in model_state:
                    model_state[prefix + key] = value
                    loaded += 1
                    break
        if loaded != len(saved):
            raise KeyError(
                f"{len(saved) - loaded} of the {len(saved)} weights in {lora_path} "
                "match no parameter of the model"
            )
        model.load_state_dict(model_state, strict=False)
        logger.info(f"Loaded {loaded}/{len(saved)} weights from {lora_path}")
