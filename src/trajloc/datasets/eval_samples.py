"""The clips an inference run generates, in order, with their output index.

A sample's index ``i`` names its output file (``<i:04d>_<stem>.mp4``) and
seeds its generation (``seed + i``), so the enumeration is part of the
results. Per evaluation set: clips with both a text-encoder cache and a VAE
latent, in sorted order, optionally filtered by ``stems`` and cut to
``max_samples``.

Sharding keeps every ``num_shards``-th sample of the list flattened across
sets. The backbones number shards differently (:class:`ShardNumbering`); a
sample's number is its seed offset, so each backbone keeps its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from trajloc.datasets.eval_sets import ResolvedEvalSet, resolve_eval_sets
from trajloc.datasets.layout import BackboneName

logger = logging.getLogger(__name__)


class ShardNumbering(Enum):
    """How a sharded run numbers its samples."""

    GLOBAL = "global"  # a sample keeps its unsharded index (Wan)
    PER_SHARD = "per_shard"  # samples are renumbered from 0 within each (shard, set) (CogVideoX)


@dataclass(frozen=True)
class EvalSample:
    """One clip to generate. ``index`` names the output and offsets the
    seed (see the module docstring)."""

    eval_set: str
    stem: str
    index: int
    text_cache_path: str
    latent_path: str
    video_path: str | None  # None if the GT video is missing


def enumerate_eval_samples(
    backbone: BackboneName,
    project_root: str | Path,
    names: list[str] | None,
    *,
    numbering: ShardNumbering,
    max_samples: int | None = None,
    stems: list[str] | None = None,
    shard: int = 0,
    num_shards: int = 1,
) -> list[EvalSample]:
    """All samples of this shard, in generation order.

    Raises ``ValueError`` if some of ``stems`` are in none of the sets, and
    ``FileNotFoundError`` if the sets have no samples at all (for example,
    when their caches aren't built or linked under ``data/``).
    """
    resolved = resolve_eval_sets(backbone, project_root, names=names)
    stems_set = set(stems) if stems else None
    available = {s.name: _eval_set_samples(s, stems_set) for s in resolved}
    if stems_set is not None:  # checked before --max_samples cuts the sets
        unknown = stems_set - {clip.stem for clips in available.values() for clip in clips}
        if unknown:
            raise ValueError(
                f"--stems not found in {[s.name for s in resolved]} for {backbone}: "
                f"{sorted(unknown)}"
            )
    by_set = {name: samples[:max_samples] for name, samples in available.items()}
    for name, samples in by_set.items():
        logger.info(f"  {name}: {len(samples)} samples")
    if not any(by_set.values()):
        raise FileNotFoundError(
            f"no {backbone} evaluation samples in {[s.text_cache_dir for s in resolved]}: build "
            f"the evaluation sets and link them under data/ (scripts/prepare_data.sh)"
        )

    if numbering is ShardNumbering.GLOBAL:
        rows = [(s.name, clip, i) for s in resolved for i, clip in enumerate(by_set[s.name])]
        if num_shards > 1:
            rows = [r for idx, r in enumerate(rows) if idx % num_shards == shard]
            logger.info(f"Shard {shard}/{num_shards}: {len(rows)} samples")
    else:
        if num_shards > 1:
            flat = [(s.name, clip) for s in resolved for clip in by_set[s.name]]
            mine = [item for idx, item in enumerate(flat) if idx % num_shards == shard]
            by_set = {s.name: [] for s in resolved}
            for name, clip in mine:
                by_set[name].append(clip)
            logger.info(
                f"  Shard {shard}/{num_shards}: {len(mine)} samples (per set: "
                f"{ {n: len(v) for n, v in by_set.items()} })"
            )
        rows = [(s.name, clip, i) for s in resolved for i, clip in enumerate(by_set[s.name])]

    return [
        EvalSample(
            eval_set=name,
            stem=clip.stem,
            index=i,
            text_cache_path=clip.text_cache_path,
            latent_path=clip.latent_path,
            video_path=clip.video_path,
        )
        for name, clip, i in rows
    ]


@dataclass(frozen=True)
class _Clip:
    """An evaluation clip's files (an :class:`EvalSample` before numbering)."""

    stem: str
    text_cache_path: str
    latent_path: str
    video_path: str | None


def _eval_set_samples(eval_set: ResolvedEvalSet, stems_set: set[str] | None) -> list[_Clip]:
    cache_dir = Path(eval_set.text_cache_dir)
    latent_dir = Path(eval_set.latent_dir)
    if not cache_dir.exists():
        logger.warning(f"text cache dir not found: {cache_dir}; skipping {eval_set.name}")
        return []
    if not latent_dir.exists():
        logger.warning(f"latent dir not found: {latent_dir}; skipping {eval_set.name}")
        return []

    # With explicit stems, look the files up by path: a large directory is
    # slow to list.
    if stems_set is not None:
        cache_files = sorted(cache_dir / f"{stem}.pt" for stem in stems_set)
        cache_files = [f for f in cache_files if f.exists()]
        latent_lookup = {
            s: latent_dir / f"{s}.pt" for s in stems_set if (latent_dir / f"{s}.pt").exists()
        }
    else:
        cache_files = sorted(cache_dir.glob("*.pt"))
        latent_lookup = {latent_file.stem: latent_file for latent_file in latent_dir.glob("*.pt")}

    videos_dir = Path(eval_set.videos_dir)
    samples: list[_Clip] = []
    for cache_file in cache_files:
        latent_file = latent_lookup.get(cache_file.stem)
        if latent_file is None:
            continue
        video_path = videos_dir / f"{cache_file.stem}.mp4"
        samples.append(
            _Clip(
                stem=cache_file.stem,
                text_cache_path=str(cache_file),
                latent_path=str(latent_file),
                video_path=str(video_path) if video_path.exists() else None,
            )
        )
    return samples
