"""Logging set-up for the command-line entry points.

Library modules log through ``logging.getLogger(__name__)`` and never print;
every CLI calls :func:`setup_logging` once, so all output shares one format.
"""

from __future__ import annotations

import logging

FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Send log records at ``level`` and above to stderr, in :data:`FORMAT`."""
    logging.basicConfig(level=level, format=FORMAT)
