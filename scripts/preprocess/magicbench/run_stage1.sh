#!/usr/bin/env bash
# MagicBench Stage 1 for one shard of the clips: 1A tracking (SAM 2 + CoTracker3),
# then 1B naming and prompts (Qwen2.5-VL). Both steps skip the clips they have
# already done.
#
# Usage (from the repository root):
#   TRAJLOC_MAGICBENCH_ROOT=<MagicBench> GPU=0 SHARD=0 NUM_SHARDS=1 bash scripts/preprocess/magicbench/run_stage1.sh
# Env: OUT_ROOT (runs/preprocess/magicbench_base), TRAJLOC_SAM2_CHECKPOINT
#      (checkpoints/sam2.1_hiera_large.pt).
set -euo pipefail
: "${TRAJLOC_MAGICBENCH_ROOT:?set TRAJLOC_MAGICBENCH_ROOT}"

OUT_ROOT="${OUT_ROOT:-runs/preprocess/magicbench_base}"
GPU="${GPU:-0}"
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE/../../../src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$GPU"
export TRAJLOC_SAM2_CHECKPOINT="${TRAJLOC_SAM2_CHECKPOINT:-$HERE/../../../checkpoints/sam2.1_hiera_large.pt}"
[ -f "$TRAJLOC_SAM2_CHECKPOINT" ] || { echo "SAM 2.1 checkpoint not found: $TRAJLOC_SAM2_CHECKPOINT" >&2; exit 1; }

mapfile -t STEMS < <(find "$TRAJLOC_MAGICBENCH_ROOT/videos" -maxdepth 1 -name '*.mp4' -printf '%f\n' \
    | sed 's/\.mp4$//' | sort -u | awk -v i="$SHARD" -v n="$NUM_SHARDS" 'NR % n == i')
[ "${#STEMS[@]}" -gt 0 ] || { echo "no clips for shard $SHARD/$NUM_SHARDS in $TRAJLOC_MAGICBENCH_ROOT/videos" >&2; exit 1; }
mkdir -p "$OUT_ROOT/logs"
echo "# MagicBench Stage 1: shard $SHARD/$NUM_SHARDS, ${#STEMS[@]} clips -> $OUT_ROOT"

python -u "$HERE/track_objects.py" --stems "${STEMS[@]}" --magicbench_root "$TRAJLOC_MAGICBENCH_ROOT" --out_root "$OUT_ROOT" \
    --device cuda:0 2>&1 | tee -a "$OUT_ROOT/logs/tracking_shard$SHARD.log"

python -u "$HERE/name_objects.py" --stems "${STEMS[@]}" --out_root "$OUT_ROOT" \
    --device cuda:0 2>&1 | tee -a "$OUT_ROOT/logs/naming_shard$SHARD.log"
