#!/usr/bin/env python3
"""SA-V Stage 1: build clips (video, trajectories, prompt) from SA-V + SVG2.

    python scripts/preprocess/sav/extract_clips.py --sav_root <SA-V> \
        --out_root runs/preprocess/sav_base --shard 0 --num_shards 8

Uses CoTracker3 and the SVG2 parquet files.
Resumable. ``run_stage1.sh`` launches one shard per GPU.
See :mod:`trajloc.preprocess.stage1.sav.extract`.
"""

import argparse
import os
import sys
from pathlib import Path

from trajloc.cli.arguments import (
    DefaultsHelpFormatter,
    add_shard_arguments,
    parse_sharded_args,
    stems_from_args,
)
from trajloc.preprocess.stage1.sav.extract import SHARD_SEED, SavSource, run


def main() -> None:
    p = argparse.ArgumentParser(
        description="SA-V Stage 1: build clips from SA-V + SVG2.",
        formatter_class=DefaultsHelpFormatter,
    )
    p.add_argument(
        "--sav_root",
        type=Path,
        default=os.environ.get("TRAJLOC_SAV_ROOT"),
        help="SA-V release root (train/sav_train/, svg2/); default $TRAJLOC_SAV_ROOT",
    )
    p.add_argument("--out_root", required=True, type=Path, help="the base root (written)")
    add_shard_arguments(p)
    p.add_argument("--seed", type=int, default=SHARD_SEED, help="sharding hash seed")
    p.add_argument(
        "--video_ids",
        nargs="+",
        default=None,
        help="process exactly these videos, space- or comma-separated (overrides sharding)",
    )
    p.add_argument(
        "--max_samples", type=int, default=0, help="first N videos of the shard (0: all)"
    )
    p.add_argument("--device", default="cuda:0", help="torch device")
    args = parse_sharded_args(p)
    if args.sav_root is None:
        p.error("--sav_root is required (or set TRAJLOC_SAV_ROOT)")
    failed = run(
        out_dir=args.out_root,
        source=SavSource(Path(args.sav_root)),
        shard=args.shard,
        num_shards=args.num_shards,
        seed=args.seed,
        video_ids=stems_from_args(args.video_ids),
        max_samples=args.max_samples,
        device=args.device,
    )
    if failed:
        sys.exit(1)  # the failed videos have errors/<video>.err; a rerun retries them


if __name__ == "__main__":
    main()
