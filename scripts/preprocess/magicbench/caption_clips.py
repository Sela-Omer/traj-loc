#!/usr/bin/env python3
"""MagicBench: caption every base clip with Qwen2.5-VL (into <out_root>/captions/).

    python scripts/preprocess/magicbench/caption_clips.py \
        --base_root runs/preprocess/magicbench_base --out_root runs/preprocess/magicbench \
        [--shard i --num_shards n]

Resumable; exits 1 if a clip failed (rerun to retry it). Then run
``join_captions.py``. See :mod:`trajloc.preprocess.stage1.scene_captions`.
"""

from trajloc.cli.preprocess import captions_main

if __name__ == "__main__":
    captions_main("MagicBench")
