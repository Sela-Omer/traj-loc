"""Run inference for one backbone over the evaluation sets.

Output: ``<project>/runs/infer/<backbone>/<run_name>/<eval set>/generated/
<i:04d>_<stem>.mp4``, plus ``run_config.json``. Clips whose output exists are
skipped, so an interrupted run resumes, and shards can share a run directory.
A clip's index ``i`` (its file name and seed) depends on the clip selection
and, for CogVideoX, on the sharding, so a run directory only accepts
invocations with the settings it was started with.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from trajloc.backbones import GenerationSettings, get_backbone
from trajloc.datasets.eval_samples import enumerate_eval_samples
from trajloc.datasets.eval_sets import resolve_eval_sets
from trajloc.datasets.layout import BackboneName
from trajloc.inference.video import write_video

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceConfig:
    """One inference run.

    Attributes:
        training_config_path: the training YAML the checkpoint was trained with.
        checkpoint_dir: a ``checkpoint-<step>/`` directory.
        run_name: output directory name.
        project_root: resolves ``data/`` and ``runs/``.
        guidance_scale: CFG scale (released: 5.0 Wan, 6.0 CogVideoX).
        device: torch device string.
        num_steps: sampling steps.
        seed: clip ``i`` is generated with seed ``seed + i``.
        shift: flow-matching scheduler shift (Wan only).
        max_samples, stems, eval_sets, shard, num_shards: which clips
            (see :func:`~trajloc.datasets.eval_samples.enumerate_eval_samples`).
    """

    training_config_path: str | Path
    checkpoint_dir: str | Path
    run_name: str
    project_root: str | Path
    guidance_scale: float
    device: str = "cuda:0"
    num_steps: int = 50
    seed: int = 42
    shift: float | None = None
    max_samples: int | None = None
    eval_sets: list[str] | None = None
    stems: list[str] | None = None
    shard: int = 0
    num_shards: int = 1


def run_inference(backbone: BackboneName, run: InferenceConfig) -> Path:
    """Generate every clip of this shard; returns the run directory."""
    spec = get_backbone(backbone)
    project_root = Path(run.project_root)
    run_dir = (project_root / "runs" / "infer" / backbone / run.run_name).resolve()
    # The clips first: a selection error (unknown stems, no caches) must not
    # leave a run directory behind or cost a model load.
    samples = enumerate_eval_samples(
        backbone,
        project_root,
        run.eval_sets,
        numbering=spec.shard_numbering,
        max_samples=run.max_samples,
        stems=run.stems,
        shard=run.shard,
        num_shards=run.num_shards,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run.training_config_path) as f:
        config = yaml.safe_load(f)
    _check_or_save_run_config(run_dir, run)

    generator = spec.load_generator(
        config,
        GenerationSettings(
            checkpoint_dir=run.checkpoint_dir,
            device=run.device,
            guidance_scale=run.guidance_scale,
            num_steps=run.num_steps,
            shift=run.shift,
        ),
    )

    for eval_set in resolve_eval_sets(backbone, project_root, names=run.eval_sets):
        (run_dir / eval_set.name / "generated").mkdir(parents=True, exist_ok=True)

    total = 0
    elapsed_total = 0.0
    failed: list[str] = []
    for sample in samples:
        name = f"{sample.index:04d}_{sample.stem}"
        gen_path = run_dir / sample.eval_set / "generated" / f"{name}.mp4"
        if gen_path.exists():
            logger.info(f"[{sample.eval_set}] {name}: exists; skipping")
            continue
        t0 = time.time()
        try:
            frames = generator.generate(sample, seed=run.seed + sample.index)
            write_video(gen_path, frames)
        except Exception:
            logger.exception(f"[{sample.eval_set}] {name}: failed")
            failed.append(f"{sample.eval_set}/{name}")
            continue
        elapsed = time.time() - t0
        elapsed_total += elapsed
        total += 1
        logger.info(f"[{sample.eval_set}] {name}: {elapsed:.1f}s")

    logger.info(
        f"Done: {total} videos in {elapsed_total:.0f}s "
        f"({elapsed_total / max(total, 1):.1f}s/sample). Output: {run_dir}"
    )
    if failed:
        raise RuntimeError(f"{len(failed)} clip(s) failed (see the log): {', '.join(failed)}")
    return run_dir


# The run settings that don't decide which clips a shard generates or where
# it runs. Every other setting fixes the clips' numbering, seeds or sampling,
# so all shards and resumes of a run must agree on it.
_PER_INVOCATION_SETTINGS = ("run_name", "project_root", "device", "shard")


def _check_or_save_run_config(run_dir: Path, run: InferenceConfig) -> None:
    """Write ``run_config.json``, or, if the run directory already has one
    (a resume, or another shard), refuse settings that differ from it."""
    payload = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in asdict(run).items()
        if k not in _PER_INVOCATION_SETTINGS
    }
    path = run_dir / "run_config.json"
    if path.exists():
        saved = json.loads(path.read_text())
        changed = {k: (saved.get(k), v) for k, v in payload.items() if saved.get(k) != v}
        if changed:
            details = "; ".join(f"{k}: {old!r} -> {new!r}" for k, (old, new) in changed.items())
            raise ValueError(
                f"{run_dir} was generated with other settings ({details}). The clips' numbering "
                f"and seeds depend on them, so use a new --run_name."
            )
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)
