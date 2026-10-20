#!/usr/bin/env bash
# SA-V Stage 1 over several GPUs: one extract_clips.py shard per GPU, logging
# to <OUT_ROOT>/logs/; waits for all and fails if any shard failed. Resumable:
# re-run to continue.
#
# Usage (from the repository root):
#   TRAJLOC_SAV_ROOT=<SA-V release> GPUS=0,1,2,3 bash scripts/preprocess/sav/run_stage1.sh
# Env: OUT_ROOT (runs/preprocess/sav_base).
set -euo pipefail
: "${TRAJLOC_SAV_ROOT:?set TRAJLOC_SAV_ROOT to the SA-V release root}"
OUT_ROOT="${OUT_ROOT:-runs/preprocess/sav_base}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE/../../../src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

mkdir -p "$OUT_ROOT/logs"
IFS=',' read -r -a gpu_arr <<< "$GPUS"
pids=()
for idx in "${!gpu_arr[@]}"; do
    gpu="${gpu_arr[$idx]}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u "$HERE/extract_clips.py" \
        --sav_root "$TRAJLOC_SAV_ROOT" --out_root "$OUT_ROOT" \
        --shard "$idx" --num_shards "${#gpu_arr[@]}" --device cuda:0 \
        >> "$OUT_ROOT/logs/stage1_shard${idx}.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "# SA-V Stage 1 done: failed shards=$fail, clips=$(find "$OUT_ROOT/videos" -maxdepth 1 -name "*.mp4" 2>/dev/null | wc -l)"
[ "$fail" -eq 0 ]
