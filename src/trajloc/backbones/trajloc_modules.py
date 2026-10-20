"""``trajloc_modules.safetensors``: TrajLoc's own modules in a checkpoint.

A TrajLoc checkpoint directory, of either backbone, holds::

    <lora file>                    the backbone's LoRA weights (see the backbone)
    trajloc_modules.safetensors    the modules TrajLoc adds, flattened:
                                     supervisor.<name>          the attention supervisor
                                     appearance_encoder.<name>  the appearance encoder
    config.yaml                    the training config (what inference rebuilds the model from)

A released model also ships ``trajectory_encoder.safetensors``, the frozen
trajectory encoder. This module reads and writes ``trajloc_modules.safetensors``
and writes ``config.yaml`` (the commands read it as ``<checkpoint>/config.yaml``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from safetensors.torch import load_file, save_file
from torch import nn

from trajloc.core.config import RUN_KEYS
from trajloc.core.supervision import AttentionSupervisor

logger = logging.getLogger(__name__)

TRAJLOC_MODULES_FILE = "trajloc_modules.safetensors"
CONFIG_FILE = "config.yaml"


def save_trajloc_modules(
    checkpoint_dir: str | Path, supervisor: AttentionSupervisor, appearance_encoder: nn.Module
) -> Path:
    """Write ``checkpoint_dir/trajloc_modules.safetensors``; returns its path."""
    modules = {"supervisor": supervisor, "appearance_encoder": appearance_encoder}
    state = {
        f"{name}.{key}": value.detach().contiguous()
        for name, module in modules.items()
        for key, value in module.state_dict().items()
    }
    path = Path(checkpoint_dir) / TRAJLOC_MODULES_FILE
    save_file(state, str(path))
    logger.info(f"saved TrajLoc's modules to {path}")
    return path


def load_trajloc_modules(
    checkpoint_dir: str | Path, supervisor: AttentionSupervisor, appearance_encoder: nn.Module
) -> None:
    """Load ``checkpoint_dir/trajloc_modules.safetensors`` into the modules,
    strictly: every parameter and buffer must be there, and nothing else. A
    missing file is an error: a model must never run on freshly initialized
    modules."""
    path = Path(checkpoint_dir) / TRAJLOC_MODULES_FILE
    if not path.exists():
        raise FileNotFoundError(f"no TrajLoc modules at {path}")
    state = load_file(str(path))
    modules = {"supervisor": supervisor, "appearance_encoder": appearance_encoder}
    for name, module in modules.items():
        prefix = f"{name}."
        module.load_state_dict(
            {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        )
    unknown = [k for k in state if not k.startswith(tuple(f"{n}." for n in modules))]
    if unknown:
        raise KeyError(f"{path}: unknown keys {unknown[:5]}")
    logger.info(f"loaded TrajLoc's modules from {path} (sigma={supervisor.sigma.item():.4f})")


def save_checkpoint_config(checkpoint_dir: str | Path, config: dict[str, Any]) -> Path:
    """Write the training ``config`` (without its run-specific keys) as
    ``checkpoint_dir/config.yaml``."""
    path = Path(checkpoint_dir) / CONFIG_FILE
    with open(path, "w") as f:
        yaml.safe_dump({k: v for k, v in config.items() if k not in RUN_KEYS}, f, sort_keys=False)
    return path
