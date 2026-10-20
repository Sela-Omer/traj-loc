"""The shared front ends of the preprocessing scripts (``scripts/preprocess/``).

- :func:`captions_main`: ``scripts/preprocess/<dataset>/caption_clips.py``;
- :func:`benchmark_join_main`: ``scripts/preprocess/{movebench,magicbench}/join_captions.py``;
- :func:`stage2_main`: the Stage-2 steps' scripts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from trajloc.cli.arguments import add_shard_arguments, parse_sharded_args, parser
from trajloc.logs import setup_logging


def captions_main(dataset: str) -> None:
    """``scripts/preprocess/<dataset>/caption_clips.py``: caption a base root's
    clips with Qwen2.5-VL into the dataset root's ``captions/``. Exits 1 if
    any clip failed (it has no record, so a rerun retries it)."""
    from trajloc.datasets.layout import DatasetRoot
    from trajloc.preprocess.stage1.qwen import QWEN_MODEL_ID
    from trajloc.preprocess.stage1.scene_captions import caption_clips

    p = parser(f"Caption every {dataset} base clip with Qwen2.5-VL.")
    p.add_argument("--base_root", required=True, type=Path, help="the base root, with videos/")
    p.add_argument(
        "--out_root", required=True, type=Path, help="the dataset root; captions/ goes here"
    )
    add_shard_arguments(p)
    p.add_argument("--max_samples", type=int, default=0, help="at most N new captions (0: all)")
    p.add_argument("--device", default="cuda:0", help="torch device")
    p.add_argument("--model_path", default=QWEN_MODEL_ID, help="Qwen2.5-VL hub id or local path")
    args = parse_sharded_args(p)
    videos_dir = DatasetRoot(args.base_root.resolve()).videos_dir
    if not videos_dir.is_dir():
        p.error(f"videos/ not found: {videos_dir}")
    failed = caption_clips(
        videos_dir,
        DatasetRoot(args.out_root.resolve()).captions_dir,
        shard=args.shard,
        num_shards=args.num_shards,
        max_samples=args.max_samples,
        device=args.device,
        model_path=args.model_path,
    )
    if failed:
        sys.exit(1)


def benchmark_join_main(dataset: str, *, captions_dir: str | None) -> None:
    """``scripts/preprocess/<dataset>/join_captions.py``: build a benchmark's
    dataset root, each object prompt followed by the scene caption (read from
    ``<out_root>/<captions_dir>/``, or with ``None`` the benchmark's own
    caption in the base records). Links what the root shares with the base
    root first. Exits 1 if any clip failed."""
    from trajloc.preprocess.stage1.caption_join import join_benchmark_captions, link_shared_layout
    from trajloc.preprocess.stage1.errors import Stage1Error

    p = parser(f"Build the {dataset} dataset root: object prompts + scene captions.")
    p.add_argument("--base_root", required=True, type=Path, help="the base root (Stage 1)")
    p.add_argument("--out_root", required=True, type=Path, help="the dataset root (written)")
    p.add_argument(
        "--steps", default="all", choices=["all", "links", "prompts"], help="which steps to run"
    )
    args = p.parse_args()
    setup_logging()
    base_root, out_root = args.base_root.resolve(), args.out_root.resolve()
    if not base_root.is_dir():
        p.error(f"base root not found: {base_root}")
    rc = 0
    try:
        if args.steps in ("all", "links"):
            link_shared_layout(base_root, out_root)
        if args.steps in ("all", "prompts"):
            captions = out_root / captions_dir if captions_dir is not None else None
            rc = join_benchmark_captions(base_root, out_root, captions_dir=captions)
    except Stage1Error as e:
        sys.exit(f"error: {e}")
    sys.exit(rc)


def stage2_main(
    description: str,
    make_step: Callable[[argparse.Namespace], Any],
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    """A Stage-2 step's CLI: ``--root``, the step's own arguments
    (``add_arguments``), sharding, ``--max_samples`` and ``--device``; then
    ``make_step(args).run(...)``."""
    p = parser(description)
    p.add_argument("--root", required=True, type=Path, help="Stage-1 dataset root")
    if add_arguments is not None:
        add_arguments(p)
    add_shard_arguments(p)
    p.add_argument("--max_samples", type=int, default=0, help="first N clips of the shard (0: all)")
    p.add_argument("--device", default="cuda:0", help="torch device")
    args = parse_sharded_args(p)
    failed = make_step(args).run(
        shard=args.shard, num_shards=args.num_shards, max_samples=args.max_samples
    )
    if failed:
        logging.getLogger(__name__).error(f"{failed} clip(s) failed; rerun to retry them")
        sys.exit(1)
