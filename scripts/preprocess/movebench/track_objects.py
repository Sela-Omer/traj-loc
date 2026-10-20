#!/usr/bin/env python3
"""MoveBench Stage 1A: track every object of each clip (masks, trajectories, crops).

    python scripts/preprocess/movebench/track_objects.py --movebench_root <MoveBench/en> \
        --stems <stem> ... --out_root runs/preprocess/movebench_base

Uses SAM 2 and CoTracker3. Resumable.
Then ``name_objects.py``. See :mod:`trajloc.preprocess.stage1.bench.object_tracking`.
"""

import argparse
import os
import sys
from pathlib import Path

from trajloc.cli.arguments import DefaultsHelpFormatter, add_stems_argument, stems_from_args
from trajloc.logs import setup_logging
from trajloc.preprocess.stage1.bench.benchmarks import MoveBench
from trajloc.preprocess.stage1.bench.object_tracking import run


def main() -> None:
    p = argparse.ArgumentParser(
        description="MoveBench Stage 1A: track every object of each clip.",
        formatter_class=DefaultsHelpFormatter,
    )
    p.add_argument(
        "--movebench_root",
        type=Path,
        default=os.environ.get("TRAJLOC_MOVEBENCH_ROOT"),
        help="MoveBench/en (default: $TRAJLOC_MOVEBENCH_ROOT)",
    )
    add_stems_argument(p, "clip stems to process", required=True)
    p.add_argument("--out_root", required=True, type=Path, help="the base root (written)")
    p.add_argument("--device", default="cuda:0", help="torch device")
    args = p.parse_args()
    setup_logging()
    if args.movebench_root is None:
        p.error("--movebench_root is required (or set TRAJLOC_MOVEBENCH_ROOT)")
    if not Path(args.movebench_root).is_dir():
        p.error(f"--movebench_root: no directory at {args.movebench_root}")
    done, failed = run(
        stems_from_args(args.stems),
        args.out_root,
        MoveBench(Path(args.movebench_root)),
        device=args.device,
    )
    if failed and not done:  # some clips failing (see errors/) leaves the rest usable
        sys.exit(f"all {failed} clips failed (see errors/)")


if __name__ == "__main__":
    main()
