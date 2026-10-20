"""Run training for one backbone.

Output: ``<project>/runs/train/<backbone>/<run_name>/`` with
``checkpoints/checkpoint-<step>/``, ``logs/`` and ``run_config.json`` (the
config as run, overrides applied).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from trajloc.backbones import get_backbone
from trajloc.datasets.layout import BackboneName

logger = logging.getLogger(__name__)


def run_training(
    backbone: BackboneName,
    config_path: str | Path,
    run_name: str,
    *,
    project_root: str | Path,
    train_steps: int | None = None,
    batch_size: int | None = None,
    checkpointing_steps: int | None = None,
) -> Path:
    """Train ``backbone`` with the YAML at ``config_path``; returns the run dir.

    ``train_steps``, ``batch_size`` and ``checkpointing_steps`` replace the
    config's values when given.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    overrides = {
        "train_steps": train_steps,
        "batch_size": batch_size,
        "checkpointing_steps": checkpointing_steps,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})

    run_dir = (Path(project_root) / "runs" / "train" / backbone / run_name).resolve()
    spec = get_backbone(backbone)
    resume = config.get("resume_from_checkpoint")
    if resume and not spec.supports_resume:
        raise ValueError(
            f"{spec.display_name} training can't resume: remove resume_from_checkpoint"
        )
    if resume and not Path(resume).is_dir():
        raise FileNotFoundError(f"resume_from_checkpoint: no checkpoint directory at {resume}")
    existing = sorted((run_dir / "checkpoints").glob("checkpoint-*"), key=_checkpoint_step)
    if existing and not resume:
        hint = ", or set resume_from_checkpoint to continue it" if spec.supports_resume else ""
        raise FileExistsError(
            f"{run_dir} already has checkpoints (the latest: {existing[-1].name}): "
            f"use a new --run_name{hint}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    config["output_dir"] = str(run_dir / "checkpoints")

    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Training {backbone}: config={config_path}, output={run_dir}")
    spec.train(config, run_dir)  # checkpoints go to config["output_dir"]
    return run_dir


def _checkpoint_step(path: Path) -> int:
    """The step of a ``checkpoint-<step>`` directory (-1 if not a number)."""
    step = path.name.rpartition("-")[2]
    return int(step) if step.isdigit() else -1
