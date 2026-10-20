"""The loop every Stage-2 step shares (Template Method).

A step turns each Stage-1 clip of a dataset root into one cached ``.pt``
payload, in ``<root>/cache/<backbone>/<cache_kind>/<stem>.pt``. :meth:`Stage2Step.run`
does everything that is common: select this shard's pending clips, load the
models once (:meth:`~Stage2Step.setup`, only if there is work), process each
clip (:meth:`~Stage2Step.process`), save atomically, and report progress.
Per-clip errors don't stop the step: they are counted and logged (the first
five in full), and :meth:`Stage2Step.run` returns their number, so the CLI
exits non-zero when any clip failed.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import torch

from trajloc.datasets.layout import BackboneName, CacheKind, DatasetRoot
from trajloc.preprocess.io import atomic_torch_save
from trajloc.preprocess.sharding import shard_items

logger = logging.getLogger(__name__)

Payload = dict[str, Any]


class Stage2Step(ABC):
    """One Stage-2 step over a dataset root, for one backbone.

    Subclasses set :attr:`cache_kind` and implement :meth:`setup` and
    :meth:`process`.
    """

    cache_kind: ClassVar[CacheKind]  # the cache directory: <root>/cache/<backbone>/<cache_kind>/
    reported_clips: ClassVar[int] = 3  # clips traced by report()
    progress_every: ClassVar[int] = 50  # clips between progress lines
    logged_errors: ClassVar[int] = 5  # clip errors logged in full

    def __init__(self, root: Path, backbone: BackboneName, *, device: str = "cuda:0") -> None:
        self.dataset_root = DatasetRoot(Path(root))
        self.backbone = backbone
        self.device = device

    @property
    def name(self) -> str:
        """The step's name in logs, e.g. ``latents[wan]``."""
        return f"{self.cache_kind}[{self.backbone}]"

    @abstractmethod
    def setup(self) -> None:
        """Load the models. Called once, before the first clip."""

    @abstractmethod
    def process(self, stem: str) -> Payload | None:
        """The payload to cache for one clip, or ``None`` if the clip can't be
        cached (the step logs why; the clip counts as failed)."""

    def report(self, stem: str, payload: Payload) -> None:  # noqa: B027 (optional hook)
        """A human-readable trace of one result (for the first ``reported_clips``)."""

    def run(self, *, shard: int = 0, num_shards: int = 1, max_samples: int = 0) -> int:
        """Process the pending clips of this shard; returns how many failed.

        Clips are the stems present in all four Stage-1 dirs, sorted, then
        this shard's (every ``num_shards``-th), then the first
        ``max_samples`` (0: all). Clips whose output exists are skipped.
        """
        out_dir = self.dataset_root.cache_dir(self.backbone, self.cache_kind)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"{self.name}: root={self.dataset_root.path} out={out_dir}")

        all_stems = self.dataset_root.stems()
        stems = shard_items(all_stems, shard, num_shards)
        if max_samples > 0:
            stems = stems[:max_samples]
        todo, skipped = _pending(stems, out_dir)
        logger.info(
            f"stems={len(all_stems)} shard={shard}/{num_shards} todo={len(todo)} skipped={skipped}"
        )
        if not todo:
            return 0

        self.setup()
        done = errs = 0
        t_start = time.time()
        for i, stem in enumerate(todo):
            try:
                t0 = time.time()
                payload = self.process(stem)
                if payload is None:
                    errs += 1
                    logger.error(f"  FAILED {stem}: no payload (see the message above)")
                    continue
                atomic_torch_save(payload, out_dir / f"{stem}.pt")
                done += 1
                elapsed = time.time() - t0
                if done <= self.reported_clips:
                    self.report(stem, payload)
                if (i + 1) % self.progress_every == 0 or (i + 1) == len(todo):
                    wall = time.time() - t_start
                    rate = done / wall if wall > 0 else 0.0
                    eta = (len(todo) - i - 1) / rate / 60 if rate > 0 else 0.0
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    logger.info(
                        f"  [{shard}] {i + 1}/{len(todo)} done={done} err={errs} "
                        f"{elapsed:.2f}s/clip ETA {eta:.0f}min peak_gpu={peak:.1f}GB"
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                errs += 1
                logger.error(f"  OOM on {stem}")
            except Exception as e:  # reported per clip; the step goes on
                errs += 1
                if errs <= self.logged_errors:
                    logger.exception(f"  ERROR {stem}: {type(e).__name__}: {e}")
                else:
                    logger.error(f"  ERROR {stem}: {type(e).__name__}: {e}")

        logger.info(
            f"DONE {self.name} shard={shard}: done={done} err={errs} "
            f"wall={(time.time() - t_start) / 60:.1f}min"
        )
        return errs


def _pending(stems: list[str], out_dir: Path) -> tuple[list[str], int]:
    """The stems without an output yet, and how many have one."""
    todo = []
    skipped = 0
    for s in stems:
        if (out_dir / f"{s}.pt").exists():
            skipped += 1
        else:
            todo.append(s)
    return todo, skipped
