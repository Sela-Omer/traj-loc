#!/usr/bin/env bash
# Download the datasets TrajLoc is trained and evaluated on, into datasets/.
#
#   scripts/download_data.sh <sav | movebench | magicbench | all>
#
#   sav         SA-V's training videos (Meta) and their SVG2 scene graphs and masks.
#               SA-V needs Meta's license: request it at
#               https://ai.meta.com/datasets/segment-anything-video-downloads/
#               and pass the links file you receive as SAV_LINKS=<file>.
#   movebench   MoveBench (English release), from Hugging Face.
#   magicbench  MagicBench, from Hugging Face.
#
# Env: DATA_DIR (datasets, under the repository root). Downloads resume, and
# an archive is unpacked into a temporary directory and moved into place only
# once complete, so a rerun after an interruption picks up where it stopped.
set -euo pipefail
dataset="${1:-}"
case "$dataset" in
    sav | movebench | magicbench | all) ;;
    *) echo "usage: $0 <sav | movebench | magicbench | all>" >&2; exit 2 ;;
esac
if [ -n "${SAV_LINKS:-}" ]; then SAV_LINKS="$(realpath "$SAV_LINKS")"; fi
cd "$(dirname "$0")/.."    # the repository root
DATA_DIR="$(realpath -m "${DATA_DIR:-datasets}")"

# The Hugging Face datasets at the revisions the release was built from.
MOVEBENCH_REVISION=d02e0c4d74abcc1af9f4913815b76348511f74b3
MAGICBENCH_REVISION=12284d1d82d930f168a87a58e77c37b3db53ae0d
SVG2_REVISION=118bab8ec8b3101099060f94062dabf30093c6ba

# wget's progress bar on a terminal; one line per file in a log.
if [ -t 2 ]; then WGET_PROGRESS=(-q --show-progress); else WGET_PROGRESS=(-nv); fi

need() { command -v "$1" >/dev/null || { echo "$1 not found: $2" >&2; exit 1; }; }
sav_links() { tr -d '\r' < "$SAV_LINKS"; }    # the links file, with any Windows line endings removed
need hf "activate the TrajLoc environment (scripts/env/install.sh)"
case "$dataset" in
    sav | all)
        need wget "install wget"
        : "${SAV_LINKS:?set SAV_LINKS to the SA-V links file from Meta (see the header of this script)}"
        [ -f "$SAV_LINKS" ] || { echo "SAV_LINKS: no file at $SAV_LINKS" >&2; exit 1; }
        sav_links | grep -qE '^sav_[0-9]+\.tar[[:space:]]' \
            || { echo "SAV_LINKS: no sav_NNN.tar entries in $SAV_LINKS" >&2; exit 1; }
        ;;
esac
case "$dataset" in magicbench | all) need unzip "install unzip" ;; esac

unpack() {    # <archive> <destination>: extract, then move each top-level entry into place (one dataset per archive)
    local archive=$1 dest=$2 tmp
    rm -rf "${dest:?}"/.unpack.*    # left by an interrupted unpack
    tmp="$(mktemp -d "$dest/.unpack.XXXXXX")"
    case "$archive" in
        *.zip) unzip -q "$archive" -d "$tmp" ;;
        *) tar -xf "$archive" -C "$tmp" ;;
    esac
    for entry in "$tmp"/* "$tmp"/.[!.]*; do
        [ -e "$entry" ] || continue
        rm -rf "${dest:?}/$(basename "$entry")"
        mv "$entry" "$dest/"
    done
    rmdir "$tmp"
    rm "$archive"
}

sav() {
    local root="$DATA_DIR/sa-v"
    mkdir -p "$root/train/sav_train"
    rm -rf "${root:?}"/train/.unpack.*    # left by an interrupted unpack
    # The links file lists "<file name> <url>" per line; the training videos are sav_NNN.tar.
    sav_links | grep -E '^sav_[0-9]+\.tar[[:space:]]' | while read -r name url; do
        local shard="${name%.tar}" tmp found
        [ -f "$root/train/sav_train/$shard/.done" ] && continue
        wget "${WGET_PROGRESS[@]}" -c -O "$root/$name" "$url"
        # A shard unpacks to sav_train/sav_NNN/ (or, in some releases, sav_NNN/): move only that
        # shard's directory into place, so the shards unpacked before it are kept.
        tmp="$(mktemp -d "$root/train/.unpack.XXXXXX")"
        tar -xf "$root/$name" -C "$tmp"
        found="$(find "$tmp" -maxdepth 2 -type d -name "$shard" | head -1)"
        [ -n "$found" ] || { echo "$name: no $shard/ directory inside" >&2; exit 1; }
        rm -rf "${root:?}/train/sav_train/${shard:?}"
        mv "$found" "$root/train/sav_train/"
        rm -rf "${tmp:?}" "${root:?}/${name:?}"
        touch "$root/train/sav_train/$shard/.done"
    done

    # SVG2 (CC BY 4.0): SA-V's scene graphs and per-frame masks.
    hf download UWGZQ/Synthetic_Visual_Genome2 --repo-type dataset --revision "$SVG2_REVISION" \
        --local-dir "$root/svg2_hub" --include "data/cleaned/sav/*" --include "masks/cleaned/sav/*"
    mkdir -p "$root/svg2/masks"
    ln -sfn "../svg2_hub/data/cleaned/sav/train-00000-of-00001.parquet" "$root/svg2/scene_graphs_sav.parquet"
    for f in "$root"/svg2_hub/masks/cleaned/sav/*.parquet; do
        [ -e "$f" ] || { echo "SVG2: no mask files downloaded" >&2; exit 1; }
        ln -sfn "../../svg2_hub/masks/cleaned/sav/$(basename "$f")" "$root/svg2/masks/$(basename "$f")"
    done
    echo "SA-V: $root"
}

movebench() {
    local root="$DATA_DIR/movebench"
    if [ ! -f "$root/.done" ]; then
        mkdir -p "$root"
        hf download Ruihang/MoveBench en.tar.gz --repo-type dataset --revision "$MOVEBENCH_REVISION" \
            --local-dir "$root"
        unpack "$root/en.tar.gz" "$root"
        touch "$root/.done"
    fi
    echo "MoveBench: $root/en"
}

magicbench() {
    local root="$DATA_DIR/magicbench"
    if [ ! -f "$root/.done" ]; then
        mkdir -p "$root"
        for z in videos masks; do    # one at a time, each marked done once unpacked
            [ -f "$root/.done_$z" ] && continue
            hf download quanhaol/MagicBench "$z.zip" --repo-type dataset \
                --revision "$MAGICBENCH_REVISION" --local-dir "$root"
            unpack "$root/$z.zip" "$root"
            touch "$root/.done_$z"
        done
        touch "$root/.done"
    fi
    echo "MagicBench: $root"
}

case "$dataset" in
    sav) sav ;;
    movebench) movebench ;;
    magicbench) magicbench ;;
    all) movebench; magicbench; sav ;;
esac
