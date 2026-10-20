#!/usr/bin/env python3
"""Build the MoveBench dataset root: the base root's clips, each object prompt
followed by the caption MoveBench ships (the base record's ``source_caption``).

    python scripts/preprocess/movebench/join_captions.py \
        --base_root runs/preprocess/movebench_base --out_root runs/preprocess/movebench

Links what the dataset shares with the base root, then writes
``<out_root>/prompts/``. See :mod:`trajloc.preprocess.stage1.caption_join`.
"""

from trajloc.cli.preprocess import benchmark_join_main

if __name__ == "__main__":
    benchmark_join_main("MoveBench", captions_dir=None)
