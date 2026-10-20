#!/usr/bin/env bash
# Stage 2 for one dataset root: for each backbone, the VAE latents and/or the
# text caches, each sharded over the given GPUs (one process per GPU). Steps
# are resumable, so re-running picks up where a previous run stopped. Run it
# after Stage 1 has finished. A base root needs only the latents; the final
# root (which links the base root's latents) only the text caches.
#
# Usage (from the repository root):
#   ROOT=runs/preprocess/<dataset> STEPS=text GPUS=0,1,2,3 bash scripts/preprocess/stage2/run_all.sh
# Env: STEPS ("latents text"), BACKBONES ("cogvideox wan").
# Logs: <ROOT>/logs/stage2.log and <ROOT>/logs/stage2_<backbone>_<step>_shard<i>.log.
set -euo pipefail
: "${ROOT:?set ROOT to the dataset root}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
STEPS="${STEPS:-latents text}"
BACKBONES="${BACKBONES:-cogvideox wan}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE/../../../src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/stage2.log") 2>&1

failed=0
run_sharded() {  # <backbone> <step>
    local backbone=$1 step=$2 script
    case "$backbone" in
        cogvideox|wan) ;;
        *) echo "unknown backbone '$backbone' (expected: cogvideox wan)" >&2; exit 2 ;;
    esac
    case "$step" in
        latents) script=latents.py ;;
        text) script=text_cache.py ;;
        *) echo "unknown step '$step' (expected: latents text)" >&2; exit 2 ;;
    esac
    IFS=',' read -r -a gpu_arr <<< "$GPUS"
    echo "[$(date +%H:%M:%S)] ${step} [${backbone}] on GPUs $GPUS"
    local pids=() idx
    for idx in "${!gpu_arr[@]}"; do
        CUDA_VISIBLE_DEVICES="${gpu_arr[$idx]}" python -u "$HERE/$script" --backbone "$backbone" \
            --root "$ROOT" --shard "$idx" --num_shards "${#gpu_arr[@]}" \
            > "$ROOT/logs/stage2_${backbone}_${step}_shard$idx.log" 2>&1 &
        pids+=($!)
    done
    local p
    for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
}

for step in $STEPS; do
    for backbone in $BACKBONES; do
        run_sharded "$backbone" "$step"
    done
done
echo "[$(date +%H:%M:%S)] Stage 2 done for $ROOT: $failed shard(s) exited non-zero"
[ "$failed" -eq 0 ]
