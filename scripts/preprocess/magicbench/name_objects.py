#!/usr/bin/env python3
"""MagicBench Stage 1B: name every object with Qwen2.5-VL, write the prompts and trajectories.

    python scripts/preprocess/magicbench/name_objects.py --stems <stem> ... \
        --out_root runs/preprocess/magicbench_base

Run after ``track_objects.py`` on the same root.
See :mod:`trajloc.preprocess.stage1.bench.object_naming`.
"""

import argparse
import sys
from pathlib import Path

from trajloc.cli.arguments import DefaultsHelpFormatter, add_stems_argument, stems_from_args
from trajloc.logs import setup_logging
from trajloc.preprocess.stage1.bench.object_naming import run
from trajloc.preprocess.stage1.qwen import QWEN_MODEL_ID


def main() -> None:
    p = argparse.ArgumentParser(
        description="MagicBench Stage 1B: name every object, write the prompts and trajectories.",
        formatter_class=DefaultsHelpFormatter,
    )
    add_stems_argument(p, "clip stems to process", required=True)
    p.add_argument("--out_root", required=True, type=Path, help="the base root (Stage 1A's)")
    p.add_argument("--device", default="cuda:0", help="torch device")
    p.add_argument("--model_path", default=QWEN_MODEL_ID, help="Qwen2.5-VL hub id or path")
    args = p.parse_args()
    setup_logging()
    done, failed = run(
        stems_from_args(args.stems),
        args.out_root,
        captions={},
        device=args.device,
        model_path=args.model_path,
    )
    if failed and not done:  # some clips failing (see errors/) leaves the rest usable
        sys.exit(f"all {failed} clips failed (see errors/)")


if __name__ == "__main__":
    main()
