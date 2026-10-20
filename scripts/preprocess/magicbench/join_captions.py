#!/usr/bin/env python3
"""Build the MagicBench dataset root: the base root's clips, each object prompt
followed by the scene captions in ``<out_root>/captions/`` (``caption_clips.py``).

    python scripts/preprocess/magicbench/join_captions.py \
        --base_root runs/preprocess/magicbench_base --out_root runs/preprocess/magicbench

Links what the dataset shares with the base root, then writes
``<out_root>/prompts/``. See :mod:`trajloc.preprocess.stage1.caption_join`.
"""

from trajloc.cli.preprocess import benchmark_join_main

if __name__ == "__main__":
    benchmark_join_main("MagicBench", captions_dir="captions")
