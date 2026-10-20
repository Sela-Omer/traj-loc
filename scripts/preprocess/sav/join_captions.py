#!/usr/bin/env python3
"""Build the SA-V dataset root: the base root's clips, with a scene caption
appended to each object prompt.

    python scripts/preprocess/sav/join_captions.py \
        --base_root runs/preprocess/sav_base --out_root runs/preprocess/sav

Run after ``caption_clips.py`` has filled ``<out_root>/captions/``; the
``links`` step needs no captions and can run earlier. Then run
``normalize_captions.py --apply``. See :mod:`trajloc.preprocess.stage1.caption_join`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from trajloc.cli.arguments import DefaultsHelpFormatter, add_stems_argument, stems_from_args
from trajloc.datasets.layout import DatasetRoot
from trajloc.logs import setup_logging
from trajloc.preprocess.stage1.caption_join import join_sav_captions, link_shared_layout
from trajloc.preprocess.stage1.errors import Stage1Error

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build the SA-V dataset root: object prompts + scene captions.",
        formatter_class=DefaultsHelpFormatter,
    )
    p.add_argument("--base_root", required=True, type=Path, help="the base root (Stage 1)")
    p.add_argument("--out_root", required=True, type=Path, help="the dataset root (written)")
    p.add_argument("--workers", type=int, default=32, help="parallel prompt writers")
    p.add_argument(
        "--allow_partial",
        action="store_true",
        help="skip clips without a caption instead of failing",
    )
    p.add_argument(
        "--steps", default="all", choices=["all", "links", "prompts"], help="which steps to run"
    )
    add_stems_argument(p, "prompts step only: these stems (e.g. for a smoke test)")
    args = p.parse_args()
    setup_logging()
    base_root, out_root = args.base_root.resolve(), args.out_root.resolve()
    if not base_root.is_dir():
        p.error(f"base root not found: {base_root}")
    logger.info(f"base_root={base_root} out_root={out_root} steps={args.steps}")

    rc = 0
    try:
        if args.steps in ("all", "links"):
            link_shared_layout(base_root, out_root)
        if args.steps in ("all", "prompts"):
            rc = join_sav_captions(
                base_root,
                out_root,
                workers=args.workers,
                allow_partial=args.allow_partial,
                stems=stems_from_args(args.stems),
            )
    except Stage1Error as e:
        sys.exit(f"error: {e}")
    if args.steps in ("all", "prompts"):
        if rc:
            logger.error("the prompts step had errors; see above")
            sys.exit(rc)
        joined = {p.stem for p in DatasetRoot(out_root).prompts_dir.glob("*.json")}
        base = {p.stem for p in DatasetRoot(base_root).prompts_dir.glob("*.json")}
        logger.info(f"prompts: {len(joined)} joined, {len(base)} in the base root")
        if base - joined:
            gap = sorted(base - joined)
            logger.warning(f"{len(gap)} clips have no joined prompt yet (first 5): {gap[:5]}")


if __name__ == "__main__":
    main()
