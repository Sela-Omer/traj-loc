#!/usr/bin/env python3
"""Stage 2: a dataset root's VAE latents, for one backbone.

    python scripts/preprocess/stage2/latents.py --backbone cogvideox --root <root> \
        [--shard i --num_shards n]

Resumable. See :mod:`trajloc.preprocess.stage2.latents`.
"""

from trajloc.backbones import BACKBONES
from trajloc.cli.preprocess import stage2_main
from trajloc.preprocess.stage2.latents import LatentCacheStep


def _add_arguments(p) -> None:
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES), help="whose VAE")


def main() -> None:
    stage2_main(
        "Stage 2: a dataset root's VAE latents, for one backbone.",
        lambda args: LatentCacheStep(args.root, args.backbone, device=args.device),
        _add_arguments,
    )


if __name__ == "__main__":
    main()
