#!/usr/bin/env bash
# SA-V scene captions: one caption_clips.py shard per GPU; waits for all.
#
# Usage (from the repository root):
#   GPUS=0,1,2,3 bash scripts/preprocess/sav/run_captions.sh
# Env: BASE_ROOT (runs/preprocess/sav_base), OUT_ROOT (runs/preprocess/sav),
#      MAX_SAMPLES (0 = all, per shard).
set -euo pipefail
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
BASE_ROOT="${BASE_ROOT:-runs/preprocess/sav_base}"
OUT_ROOT="${OUT_ROOT:-runs/preprocess/sav}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE/../../../src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

mkdir -p "$OUT_ROOT/captions" "$OUT_ROOT/logs"
IFS=',' read -r -a gpu_arr <<< "$GPUS"
pids=()
for idx in "${!gpu_arr[@]}"; do
    gpu="${gpu_arr[$idx]}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u "$HERE/caption_clips.py" --base_root "$BASE_ROOT" \
        --out_root "$OUT_ROOT" --shard "$idx" --num_shards "${#gpu_arr[@]}" --device cuda:0 \
        --max_samples "$MAX_SAMPLES" > "$OUT_ROOT/logs/captions_shard${idx}.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "# captioning done: failed shards=$fail, captions=$(find "$OUT_ROOT/captions" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)"
[ "$fail" -eq 0 ]
