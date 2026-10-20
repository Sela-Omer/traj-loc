"""Weights on the Hugging Face Hub, downloaded into ``checkpoints/`` on first use.

The released TrajLoc models (one Hub repository per backbone, the backbones'
``release_repo``) and Wan's base model are fetched here; later runs find the
files in place and touch no network.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

from trajloc.paths import project_root

logger = logging.getLogger(__name__)

_REPO_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*")  # <owner>/<name>


def checkpoints_dir() -> Path:
    """Where downloads go: ``<repository>/checkpoints/``."""
    return project_root() / "checkpoints"


def is_repo_id(value: str) -> bool:
    """Whether ``value`` names a Hub repository (``<owner>/<name>``) rather
    than a local path: it has that form and no such path exists."""
    return _REPO_ID.fullmatch(value) is not None and not Path(value).exists()


def local_dir_for(repo_id: str) -> Path:
    """The download directory of ``repo_id``: ``checkpoints/<name>``."""
    return checkpoints_dir() / repo_id.split("/")[1]


def fetch(
    repo_id: str,
    local_dir: Path,
    *,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    """Download ``repo_id`` (the files matching ``allow_patterns``) into
    ``local_dir``, resuming a partial download, and return ``local_dir``.
    ``FileNotFoundError`` if the repository or revision can't be reached."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    logger.info(f"downloading {repo_id} into {local_dir} (once: later runs use the files there)")
    try:
        snapshot_download(
            repo_id,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
        )
    except (HfHubHTTPError, LocalEntryNotFoundError) as e:
        message = f"could not download {repo_id} from the Hugging Face Hub: {e}"
        raise FileNotFoundError(message) from e
    return local_dir
