#!/usr/bin/env bash
# Build a dataset for training or evaluation, end to end, from its download
# (scripts/download_data.sh) to runs/preprocess/<dataset>/.
#
#   GPUS=0,1,2,3 scripts/prepare_data.sh <sav | movebench | magicbench>
#
# Stage 1 cuts the clips, tracks every object and writes the prompts; Stage 2
# caches each model's latents and text encodings. Every step resumes where it
# stopped, so rerun the command after an interruption.
#
# Env: GPUS (the visible GPUs, CUDA_VISIBLE_DEVICES, else 0), DATA_DIR (datasets, under the repository
#      root), BACKBONES ("cogvideox wan": the models to build caches for), TRAJLOC_SAM2_CHECKPOINT
#      (checkpoints/sam2.1_hiera_large.pt).
set -euo pipefail
DATASET="${1:-}"
export GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}" BACKBONES="${BACKBONES:-cogvideox wan}"
PRE=scripts/preprocess
OUT=runs/preprocess
cd "$(dirname "$0")/.."    # the repository root
DATA_DIR="$(realpath -m "${DATA_DIR:-datasets}")"    # as scripts/download_data.sh resolves it

require_checkpoints() {    # what Stage 1 and 2 load from checkpoints/
    # The trajectory encoder is the released one; both models ship the same
    # file: a downloaded model's, else the file alone from the Hub.
    if [ ! -e checkpoints/trajectory_encoder.safetensors ]; then
        local model
        model=$(find checkpoints -maxdepth 2 -path 'checkpoints/TrajLoc-*/trajectory_encoder.safetensors' 2>/dev/null | sort | head -1)
        if [ -n "$model" ]; then
            ln -sfn "${model#checkpoints/}" checkpoints/trajectory_encoder.safetensors
        else
            local repo rev
            read -r repo rev < <(python -c "from trajloc.backbones import BACKBONES as B; \
print(B['cogvideox'].release_repo, B['cogvideox'].release_revision)")
            hf download "$repo" trajectory_encoder.safetensors --revision "$rev" \
                --local-dir checkpoints >/dev/null
        fi
    fi
    # CoTracker3 (torch.hub): fetched once here, so parallel shards don't race to download it.
    CUDA_VISIBLE_DEVICES="" python -c \
        "from trajloc.preprocess.stage1.tracking import CoTracker; CoTracker(device='cpu')"
    # SAM 2.1 (the benchmarks' masks): downloaded to a .part file, moved into place once complete.
    local sam2="${TRAJLOC_SAM2_CHECKPOINT:-checkpoints/sam2.1_hiera_large.pt}"
    if [ "$DATASET" != sav ] && [ ! -f "$sam2" ]; then
        mkdir -p "$(dirname "$sam2")"
        wget -nv -c -O "$sam2.part" \
            https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
        mv "$sam2.part" "$sam2"
    fi
}

stage2() {    # <root> <steps>
    ROOT="$1" STEPS="$2" bash "$PRE/stage2/run_all.sh"
}

benchmark_stage1() {    # <name>: one shard per GPU
    local name=$1 pids=() fail=0 i
    IFS=',' read -r -a gpus <<< "$GPUS"
    for i in "${!gpus[@]}"; do
        GPU="${gpus[$i]}" SHARD="$i" NUM_SHARDS="${#gpus[@]}" bash "$PRE/$name/run_stage1.sh" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
    [ "$fail" -eq 0 ] || { echo "$name Stage 1: $fail shard(s) failed" >&2; exit 1; }
}

link_eval_set() {    # <name>: where inference looks for it
    mkdir -p data
    ln -sfn "../$OUT/$1" "data/$1"
}

case "$DATASET" in
    sav)
        require_checkpoints
        export TRAJLOC_SAV_ROOT="${TRAJLOC_SAV_ROOT:-$DATA_DIR/sa-v}"
        bash "$PRE/sav/run_stage1.sh"
        python "$PRE/sav/build_splits.py" --root "$OUT/sav_base" --test_scenes 100
        stage2 "$OUT/sav_base" latents
        bash "$PRE/sav/run_captions.sh"
        python "$PRE/sav/join_captions.py" --base_root "$OUT/sav_base" --out_root "$OUT/sav"
        python "$PRE/sav/normalize_captions.py" --root "$OUT/sav" --apply
        stage2 "$OUT/sav" text
        ;;
    movebench)
        require_checkpoints
        export TRAJLOC_MOVEBENCH_ROOT="${TRAJLOC_MOVEBENCH_ROOT:-$DATA_DIR/movebench/en}"
        benchmark_stage1 movebench
        stage2 "$OUT/movebench_base" latents
        python "$PRE/movebench/join_captions.py" \
            --base_root "$OUT/movebench_base" --out_root "$OUT/movebench"
        stage2 "$OUT/movebench" text
        link_eval_set movebench
        ;;
    magicbench)
        require_checkpoints
        export TRAJLOC_MAGICBENCH_ROOT="${TRAJLOC_MAGICBENCH_ROOT:-$DATA_DIR/magicbench}"
        benchmark_stage1 magicbench
        stage2 "$OUT/magicbench_base" latents
        bash "$PRE/magicbench/run_captions.sh"
        python "$PRE/magicbench/join_captions.py" \
            --base_root "$OUT/magicbench_base" --out_root "$OUT/magicbench"
        stage2 "$OUT/magicbench" text
        link_eval_set magicbench
        ;;
    *) echo "usage: GPUS=0,1,... $0 <sav | movebench | magicbench>" >&2; exit 2 ;;
esac
echo "$DATASET: $OUT/$DATASET"
