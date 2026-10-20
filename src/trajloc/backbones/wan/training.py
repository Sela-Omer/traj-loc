"""Wan training loop.

Plain PyTorch: AdamW over every trainable parameter (LoRA, the appearance
encoder), gradient clipping, a checkpoint every ``checkpointing_steps``, and
a per-step loss trace (``logs/loss_trace.jsonl``: step, loss, sigma,
step_time, timestep).

Seeding: the global RNG is seeded before the model is built (LoRA and
appearance-encoder initialization) and again right before the first
DataLoader iterator is created (the shuffle order).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from trajloc.backbones.wan.checkpoint import WanCheckpointer
from trajloc.backbones.wan.config import CONFIG_KEYS, TRAINING_CONFIG_KEYS, WanModelSettings
from trajloc.backbones.wan.model import WanTrajLocModule
from trajloc.core.config import TrajLocConfig, as_number, check_config_keys
from trajloc.datasets import build_training_dataset, cached_collate

logger = logging.getLogger(__name__)

LOG_EVERY_STEPS = 10  # training-log cadence
LOG_FIRST_STEPS = 3  # the first steps are always logged


@dataclass(frozen=True)
class WanTrainingSettings:
    """The training loop's settings (the :data:`TRAINING_CONFIG_KEYS` of a
    Wan config).

    Attributes:
        seed: seeds the model's initialization and the shuffle order.
        train_steps: steps to run.
        batch_size: clips per step.
        gradient_accumulation_steps: steps per optimizer update.
        learning_rate, beta1, beta2, epsilon, weight_decay: AdamW's.
        max_grad_norm: the gradient-clipping norm.
        checkpointing_steps: a checkpoint every this many steps.
        num_workers, pin_memory: the DataLoader's.
        output_dir: where the checkpoints go (set by the training runner).
    """

    seed: int
    train_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    max_grad_norm: float
    checkpointing_steps: int
    num_workers: int
    pin_memory: bool
    output_dir: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> WanTrainingSettings:
        """The settings of a Wan config (after the runner set ``output_dir``)."""
        check_config_keys(config, CONFIG_KEYS, "Wan")
        settings = {key: config[key] for key in TRAINING_CONFIG_KEYS}
        for f in fields(cls):
            if f.type in ("int", "float"):
                kind = int if f.type == "int" else float
                settings[f.name] = as_number(settings[f.name], kind, f"Wan config: {f.name}")
        return cls(**settings, output_dir=Path(config["output_dir"]))


class _LossTrace:
    """``logs/loss_trace.jsonl``: one JSON record per step, flushed at once."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w")

    def write(self, **record) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def __enter__(self) -> _LossTrace:
        return self

    def __exit__(self, *exc) -> None:
        self._file.close()


def _build_dataloader(trajloc_config: TrajLocConfig, settings: WanTrainingSettings):
    dataset = build_training_dataset(trajloc_config, "wan", split="train")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        collate_fn=cached_collate,
        drop_last=True,
        pin_memory=settings.pin_memory,
    )
    return dataset, dataloader


def _build_optimizer(model: WanTrajLocModule, settings: WanTrainingSettings):
    """AdamW over every trainable parameter, in ``model.parameters()`` order."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=settings.learning_rate,
        betas=(settings.beta1, settings.beta2),
        eps=settings.epsilon,
        weight_decay=settings.weight_decay,
    )
    total_trainable = sum(p.numel() for p in trainable_params)
    lora_params = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()
    )
    logger.info(f"Optimizer: {total_trainable:,} total trainable params (LoRA={lora_params:,})")
    return optimizer, trainable_params


class _Batches:
    """The DataLoader's batches, epoch after epoch. The first epoch's iterator
    (its shuffle order) is created at construction."""

    def __init__(self, dataloader) -> None:
        self._dataloader = dataloader
        self._iterator = iter(dataloader)

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._dataloader)
            return next(self._iterator)


def train(config: dict, run_dir: Path) -> None:
    """Train per ``config`` (see :class:`~trajloc.backbones.base.TrainingEntryPoint`),
    on one GPU (the first visible: pick it with ``CUDA_VISIBLE_DEVICES``)."""
    settings = WanTrainingSettings.from_config(config)
    if settings.train_steps % settings.checkpointing_steps:
        last = settings.train_steps - settings.train_steps % settings.checkpointing_steps
        logger.info(
            f"train_steps={settings.train_steps} is not a multiple of checkpointing_steps="
            f"{settings.checkpointing_steps}: the last checkpoint is step {last}"
        )
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)

    trajloc_config = TrajLocConfig.from_mapping(config["trajloc"])
    logger.info("Building WanTrajLocModule...")
    model = WanTrajLocModule(trajloc_config, WanModelSettings.from_config(config))
    model.to(torch.device("cuda"))

    logger.info("Building dataset...")
    dataset, dataloader = _build_dataloader(trajloc_config, settings)
    optimizer, trainable_params = _build_optimizer(model, settings)

    checkpointer = WanCheckpointer(save_interval=settings.checkpointing_steps, config=config)
    grad_accum = settings.gradient_accumulation_steps
    logger.info(
        f"Starting training: {settings.train_steps} steps, BS={settings.batch_size}, "
        f"grad_accum={grad_accum}, dataset={len(dataset)} samples, "
        f"output={run_dir}"
    )

    torch.manual_seed(settings.seed)  # the shuffle order
    batches = _Batches(dataloader)
    model.train()

    with _LossTrace(run_dir / "logs" / "loss_trace.jsonl") as loss_trace:
        for step in range(1, settings.train_steps + 1):
            t0 = time.time()
            batch = next(batches)

            loss, timestep = model(batch, return_timestep=True)
            loss = loss / grad_accum
            loss.backward()

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, settings.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            step_time = time.time() - t0
            sigma = model.supervisor.sigma.item()
            loss_value = float(loss.item() * grad_accum)
            loss_trace.write(
                step=step, loss=loss_value, sigma=sigma, step_time=step_time, timestep=timestep
            )

            if step % LOG_EVERY_STEPS == 0 or step <= LOG_FIRST_STEPS:
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                logger.info(
                    f"Step {step}/{settings.train_steps}: loss={loss_value:.4f}, "
                    f"sigma={sigma:.2f}, time={step_time:.2f}s, mem={mem_gb:.1f}GB, t={timestep}"
                )

            if checkpointer.should_save(step):
                checkpointer.save(model, step, settings.output_dir)

    logger.info("Training complete")
