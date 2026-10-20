#!/usr/bin/env python3
"""Clean the SA-V scene captions that Wan's tokenizer would normalize, in place.

    python scripts/preprocess/sav/normalize_captions.py --root runs/preprocess/sav [--apply]

Rewrites the affected captions and joined prompts, and deletes their text
caches so Stage 2 recomputes them. Without ``--apply`` it only reports.
See :func:`trajloc.preprocess.stage1.caption_join.normalize_captions`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trajloc.cli.arguments import DefaultsHelpFormatter, add_stems_argument, stems_from_args
from trajloc.logs import setup_logging
from trajloc.preprocess.stage1.caption_join import normalize_captions


def main() -> None:
    p = argparse.ArgumentParser(
        description="Clean the SA-V scene captions that Wan's tokenizer would normalize.",
        formatter_class=DefaultsHelpFormatter,
    )
    p.add_argument("--root", required=True, type=Path, help="the joined SA-V dataset root")
    p.add_argument("--apply", action="store_true", help="write the fixes (default: report only)")
    add_stems_argument(p, "stems to check instead of scanning captions/")
    args = p.parse_args()
    setup_logging()
    normalize_captions(args.root.resolve(), apply=args.apply, stems=stems_from_args(args.stems))


if __name__ == "__main__":
    main()
