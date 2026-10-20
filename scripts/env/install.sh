#!/usr/bin/env bash
# Create the TrajLoc environment with uv (fast), from its fully pinned lock.
#
#   scripts/env/install.sh [venv_dir]     # default: .venv, in the repository root
#
# Needs git (the pinned upstream code is in git submodules) and uv
# (https://docs.astral.sh/uv/). Rerunning it reinstalls into the same venv.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"    # repository root
venv="${1:-$here/.venv}"
lock="$here/requirements/trajloc.txt"

command -v git >/dev/null || { echo "git not found: install it" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
uv venv --help | grep -q -- --managed-python \
    || { echo "this uv is too old (no --managed-python): update it, e.g. uv self update" >&2; exit 1; }

# The pinned upstream checkouts (external/) are git submodules on sys.path: first,
# so a checkout without them fails here rather than after the install.
git -C "$here" rev-parse --git-dir >/dev/null 2>&1 \
    || { echo "$here is not a git checkout: clone it with git clone --recursive" >&2; exit 1; }
git -C "$here" submodule update --init --recursive

# install.sh chooses the interpreter itself (uv's own CPython 3.10): a user's uv
# interpreter preferences would conflict with --managed-python.
unset UV_PYTHON_PREFERENCE UV_MANAGED_PYTHON UV_NO_MANAGED_PYTHON

uv venv --python 3.10 --managed-python --allow-existing "$venv"   # uv's own CPython 3.10
# unsafe-best-match: torch wheels come from the PyTorch CUDA index, the rest from PyPI.
# SAM2_BUILD_CUDA=0: SAM 2 without its optional CUDA extension, as the released
# data was built (with it, SAM 2 also fills small mask holes, so masks differ).
# --refresh-package sam-2: uv's build cache ignores the variable, so build it anew.
SAM2_BUILD_CUDA=0 uv pip install --python "$venv" --index-strategy unsafe-best-match \
    --refresh-package sam-2 -r "$lock"
# flash-attn builds against the environment's torch, so after it. Its build
# fetches a prebuilt wheel from GitHub; without access to GitHub it compiles
# from source instead (hours, and it needs the CUDA 12.4 toolkit).
uv pip install --python "$venv" flash-attn==2.7.4.post1 --no-build-isolation
uv pip install --python "$venv" --no-deps -e "$here"   # the trajloc package itself

"$venv/bin/python" -c "import cv2" 2>/dev/null || {
    echo "OpenCV can't load: install its system libraries, e.g. apt install libgl1 libglib2.0-0" >&2
    exit 1
}
echo
echo "Done. Use it with:  source $venv/bin/activate"
