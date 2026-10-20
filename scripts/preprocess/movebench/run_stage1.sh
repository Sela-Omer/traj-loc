#!/usr/bin/env bash
# MoveBench Stage 1 for one shard of the clips: 1A tracking (SAM 2 + CoTracker3),
# then 1B naming and prompts (Qwen2.5-VL). Both steps skip the clips they have
# already done.
#
# Usage (from the repository root):
#   TRAJLOC_MOVEBENCH_ROOT=<MoveBench> GPU=0 SHARD=0 NUM_SHARDS=1 bash scripts/preprocess/movebench/run_stage1.sh
# Env: OUT_ROOT (runs/preprocess/movebench_base), TRAJLOC_SAM2_CHECKPOINT
#      (checkpoints/sam2.1_hiera_large.pt).
set -euo pipefail
: "${TRAJLOC_MOVEBENCH_ROOT:?set TRAJLOC_MOVEBENCH_ROOT}"
OUT_ROOT="${OUT_ROOT:-runs/preprocess/movebench_base}"
GPU="${GPU:-0}"
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE/../../../src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$GPU"
export TRAJLOC_SAM2_CHECKPOINT="${TRAJLOC_SAM2_CHECKPOINT:-$HERE/../../../checkpoints/sam2.1_hiera_large.pt}"
[ -f "$TRAJLOC_SAM2_CHECKPOINT" ] || { echo "SAM 2.1 checkpoint not found: $TRAJLOC_SAM2_CHECKPOINT" >&2; exit 1; }

TRACK_TXT="$TRAJLOC_MOVEBENCH_ROOT/multi_track.txt"  # the multi-object subset
[ -f "$TRACK_TXT" ] || { echo "MoveBench clip list not found: $TRACK_TXT" >&2; exit 1; }
mapfile -t STEMS < <(awk -F, 'NF>=1 {sub(/\.mp4$/,"",$1); print $1}' "$TRACK_TXT" | sort -u \
    | awk -v i="$SHARD" -v n="$NUM_SHARDS" 'NR % n == i')
[ "${#STEMS[@]}" -gt 0 ] || { echo "no clips for shard $SHARD/$NUM_SHARDS in $TRACK_TXT" >&2; exit 1; }
mkdir -p "$OUT_ROOT/logs"
echo "# MoveBench Stage 1: shard $SHARD/$NUM_SHARDS, ${#STEMS[@]} clips -> $OUT_ROOT"

python -u "$HERE/track_objects.py" --stems "${STEMS[@]}" --movebench_root "$TRAJLOC_MOVEBENCH_ROOT" --out_root "$OUT_ROOT" \
    --device cuda:0 2>&1 | tee -a "$OUT_ROOT/logs/tracking_shard$SHARD.log"

python -u "$HERE/name_objects.py" --stems "${STEMS[@]}" --out_root "$OUT_ROOT" --track_txt "$TRACK_TXT" \
    --device cuda:0 2>&1 | tee -a "$OUT_ROOT/logs/naming_shard$SHARD.log"
