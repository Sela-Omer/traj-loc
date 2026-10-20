"""The project root, and putting pinned upstream checkouts on ``sys.path``.

A leaf module (no trajloc imports), so any layer can use it.

The project root is the repository this file is in, or ``$TRAJLOC_ROOT``.
The CLIs resolve ``data/`` and ``runs/`` from the script's own repository, so
set ``TRAJLOC_ROOT`` only when the package is imported from elsewhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    """The repository root (``$TRAJLOC_ROOT`` if set)."""
    return Path(os.environ.get("TRAJLOC_ROOT", Path(__file__).resolve().parents[2]))


def add_to_path(*relative_dirs: str) -> None:
    """Put ``<project>/<dir>`` first on ``sys.path`` for each directory in
    order (moving it if it is already there), so the last one ends up first.

    Raises ``ImportError`` if a directory is missing or empty (a pinned
    checkout that was never fetched: ``git submodule update --init``).
    """
    root = project_root()
    for rel in relative_dirs:
        path = root / rel
        if not path.is_dir() or not any(path.iterdir()):
            raise ImportError(
                f"{path} is missing or empty: fetch the pinned upstream code with "
                f"`git submodule update --init --recursive`"
            )
        if str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))
