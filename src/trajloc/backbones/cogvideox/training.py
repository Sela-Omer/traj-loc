"""CogVideoX training entry: the config -> upstream ``Args`` -> ``fit()``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from trajloc.core.config import check_config_keys

logger = logging.getLogger(__name__)


# The upstream trainer's arguments that TrajLoc fixes: its own data pipeline
# and model registry, which TrajLoc replaces, and settings with no effect on
# TrajLoc's training (each with the reason).
UPSTREAM_FIXED_ARGS: dict[str, Any] = {
    "model_name": "cogvideox-5b-i2v",
    "model_type": "i2v",
    "training_type": "lora",
    "data_root": ".",  # batches come from the Stage-2 caches instead
    "caption_column": "unused",
    "video_column": "unused",
    "do_validation": False,
    "validation_steps": None,
    "validation_dir": None,
    "validation_prompts": None,
    "validation_images": None,
    "validation_videos": None,
    "gen_fps": 8,
    "train_epochs": 100,  # recomputed by upstream from train_steps
    "beta3": 0.98,  # CAME's third beta; AdamW has two
    "lr_num_cycles": 1,  # cosine-with-restarts only; the schedule is constant
    "lr_power": 1.0,  # polynomial only
    "enable_slicing": True,  # VAE memory options; training reads cached latents
    "enable_tiling": True,
    "nccl_timeout": 1800,  # seconds
    "tracker_name": "trajloc-cogvideox",
}

# The settings a CogVideoX config states (all of them, see configs/cogvideox.yaml),
# as passed to upstream's ``Args``.
ARGS_KEYS = (
    "model_path",
    "train_resolution",
    "seed",
    "train_steps",
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "optimizer",
    "beta1",
    "beta2",
    "epsilon",
    "weight_decay",
    "max_grad_norm",
    "lr_scheduler",
    "lr_warmup_steps",
    "mixed_precision",
    "gradient_checkpointing",
    "rank",
    "lora_alpha",
    "target_modules",
    "checkpointing_steps",
    "checkpointing_limit",
    "num_workers",
    "pin_memory",
    "report_to",
)
CONFIG_KEYS = frozenset({*ARGS_KEYS, "trajloc"})


def build_args(config: dict[str, Any]) -> Any:
    """The upstream trainer's ``Args`` for ``config``: its :data:`ARGS_KEYS`,
    the runner's ``output_dir`` and ``resume_from_checkpoint``, and
    :data:`UPSTREAM_FIXED_ARGS`."""
    from trajloc.backbones.cogvideox.upstream import Args

    check_config_keys(config, CONFIG_KEYS, "CogVideoX")
    settings = {key: config[key] for key in ARGS_KEYS}
    settings["train_resolution"] = tuple(settings["train_resolution"])
    settings["learning_rate"] = float(settings["learning_rate"])
    return Args(
        **UPSTREAM_FIXED_ARGS,
        **settings,
        output_dir=config["output_dir"],
        resume_from_checkpoint=config.get("resume_from_checkpoint"),
    )


def train(config: dict, run_dir: Path) -> None:
    """Train per ``config`` (see :class:`~trajloc.backbones.base.TrainingEntryPoint`)."""
    args = build_args(config)
    logger.info(
        f"model={args.model_path}, train_steps={args.train_steps}, "
        f"batch_size={args.batch_size}, output={run_dir}"
    )
    from trajloc.backbones.cogvideox.trainer import CogVideoXTrajLocTrainer

    CogVideoXTrajLocTrainer(args, config).fit()
