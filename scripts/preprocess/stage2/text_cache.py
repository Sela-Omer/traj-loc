#!/usr/bin/env python3
"""Stage 2: a dataset root's text-encoder caches, for one backbone.

    python scripts/preprocess/stage2/text_cache.py --backbone cogvideox --root <root> \
        [--dataset_name <name>] [--shard i --num_shards n]

Resumable. See :mod:`trajloc.preprocess.stage2.text_cache`.
"""

from pathlib import Path

from trajloc.backbones import BACKBONES
from trajloc.cli.preprocess import stage2_main
from trajloc.preprocess.stage2.models import TRAJECTORY_ENCODER_CHECKPOINT
from trajloc.preprocess.stage2.text_cache import TextCacheStep


def _add_arguments(p) -> None:
    p.add_argument(
        "--backbone", required=True, choices=sorted(BACKBONES), help="whose text encoder"
    )
    p.add_argument(
        "--dataset_name",
        default=None,
        help="the dataset name recorded in each cache (default: the root directory's name)",
    )
    p.add_argument(
        "--trajectory_encoder",
        type=Path,
        default=TRAJECTORY_ENCODER_CHECKPOINT,
        help="the trajectory encoder's weights",
    )


def main() -> None:
    stage2_main(
        "Stage 2: a dataset root's text-encoder caches, for one backbone.",
        lambda args: TextCacheStep(
            args.root,
            args.backbone,
            dataset_name=args.dataset_name,
            device=args.device,
            trajectory_encoder_path=args.trajectory_encoder,
        ),
        _add_arguments,
    )


if __name__ == "__main__":
    main()
