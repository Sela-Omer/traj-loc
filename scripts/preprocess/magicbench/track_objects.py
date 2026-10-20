#!/usr/bin/env python3
"""MagicBench Stage 1A: track every object of each clip (masks, trajectories, crops).

    python scripts/preprocess/magicbench/track_objects.py --magicbench_root <MagicBench> \
        --stems <stem> ... --out_root runs/preprocess/magicbench_base

Uses SAM 2 and CoTracker3. Resumable.
Then ``name_objects.py``. See :mod:`trajloc.preprocess.stage1.bench.object_tracking`.
"""

import argparse
import os
import sys
from pathlib import Path

from trajloc.cli.arguments import DefaultsHelpFormatter, add_stems_argument, stems_from_args
from trajloc.logs import setup_logging
from trajloc.preprocess.stage1.bench.benchmarks import MagicBench
from trajloc.preprocess.stage1.bench.object_tracking import run


def main() -> None:
    p = argparse.ArgumentParser(
        description="MagicBench Stage 1A: track every object of each clip.",
        formatter_class=DefaultsHelpFormatter,
    )
    p.add_argument(
        "--magicbench_root",
        type=Path,
        default=os.environ.get("TRAJLOC_MAGICBENCH_ROOT"),
        help="the MagicBench release (default: $TRAJLOC_MAGICBENCH_ROOT)",
    )
    add_stems_argument(p, "clip stems to process", required=True)
    p.add_argument("--out_root", required=True, type=Path, help="the base root (written)")
    p.add_argument("--device", default="cuda:0", help="torch device")
    args = p.parse_args()
    setup_logging()
    if args.magicbench_root is None:
        p.error("--magicbench_root is required (or set TRAJLOC_MAGICBENCH_ROOT)")
    if not Path(args.magicbench_root).is_dir():
        p.error(f"--magicbench_root: no directory at {args.magicbench_root}")
    done, failed = run(
        stems_from_args(args.stems),
        args.out_root,
        MagicBench(Path(args.magicbench_root)),
        device=args.device,
    )
    if failed and not done:  # some clips failing (see errors/) leaves the rest usable
        sys.exit(f"all {failed} clips failed (see errors/)")


if __name__ == "__main__":
    main()
