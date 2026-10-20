#!/usr/bin/env python3
"""Write the scene-disjoint train/test split of an SA-V Stage-1 root.

    python scripts/preprocess/sav/build_splits.py \
        --root runs/preprocess/sav_base --test_scenes 100

See :mod:`trajloc.preprocess.stage1.sav.splits`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trajloc.logs import setup_logging
from trajloc.preprocess.stage1.sav.splits import build_splits


def main() -> None:
    p = argparse.ArgumentParser(
        description="Write the scene-disjoint train/test split of an SA-V Stage-1 root."
    )
    p.add_argument("--root", required=True, type=Path, help="Stage-1 root with videos/")
    p.add_argument("--out", type=Path, default=None, help="default: <root>/splits.json")
    p.add_argument(
        "--test_pct",
        type=float,
        default=5.0,
        help="hold out scenes with hash %% 10000 < round(pct * 100)",
    )
    p.add_argument(
        "--test_scenes",
        type=int,
        default=None,
        help="hold out exactly N scenes (overrides --test_pct); the released split is "
        "--test_scenes 100 (what scripts/prepare_data.sh passes)",
    )
    args = p.parse_args()
    setup_logging()
    out = args.out if args.out is not None else args.root / "splits.json"
    build_splits(args.root, out, test_pct=args.test_pct, test_scenes=args.test_scenes)


if __name__ == "__main__":
    main()
