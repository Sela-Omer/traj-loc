"""Filesystem helpers shared by the preprocessing steps.

Outputs are written atomically (a private temp file, then ``os.replace``), so
a crashed or concurrent writer never leaves a half-written file that a
resuming step would take as done. JSON outputs get mode 0644 (``mkstemp``
defaults to 0600).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(payload: dict, out_path: Path) -> None:
    """Write ``payload`` to ``out_path`` atomically, world-readable (0644).

    The format is ``json.dump(payload, f, ensure_ascii=False)``: one line,
    no trailing newline, non-ASCII kept verbatim.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=out_path.name + ".", dir=out_path.parent)
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.chmod(tmp, 0o644)  # mkstemp creates it 0600: a normal file's mode
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch_save(payload: Any, out_path: Path) -> None:
    """``torch.save`` to ``out_path`` atomically.

    Each writer saves into its own temporary directory, under the name
    ``<out_path name>.tmp`` (``torch.save`` records the file name in the
    archive, so the name, and so the bytes, don't depend on the writer),
    then renames the file into place.
    """
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{out_path.name}.", dir=out_path.parent))
    try:
        tmp = tmp_dir / f"{out_path.name}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, out_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
