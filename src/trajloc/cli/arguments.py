"""Argument helpers every command line shares (``trajloc`` and the scripts)."""

from __future__ import annotations

import argparse
from typing import Any

from trajloc.logs import setup_logging


class DefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Appends each option's default to its help, unless the option is
    required, has no default, or its help already states one."""

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        no_default = action.required or action.default in (None, argparse.SUPPRESS)
        if no_default or "(default" in help_text:
            return help_text
        return super()._get_help_string(action) or help_text


def parser(description: str, **kwargs: Any) -> argparse.ArgumentParser:
    """An ``ArgumentParser`` that shows each option's default in ``--help``."""
    return argparse.ArgumentParser(
        description=description, formatter_class=DefaultsHelpFormatter, **kwargs
    )


def add_shard_arguments(p: argparse.ArgumentParser) -> None:
    """``--shard`` and ``--num_shards``."""
    p.add_argument(
        "--shard",
        type=int,
        default=0,
        help="this worker's shard index, in [0, --num_shards)",
    )
    p.add_argument(
        "--num_shards", type=int, default=1, help="number of shards the clips are split into"
    )


def check_shard_arguments(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Exit with a usage error unless ``0 <= --shard < --num_shards``."""
    if args.num_shards < 1:
        p.error("--num_shards must be at least 1")
    if not 0 <= args.shard < args.num_shards:
        p.error(f"--shard must be in [0, --num_shards={args.num_shards})")


def add_stems_argument(
    p: argparse.ArgumentParser, help_text: str, *, required: bool = False
) -> None:
    """``--stems``: clip stems, space- or comma-separated (``--stems a b`` and
    ``--stems a,b`` are the same; read them with :func:`stems_from_args`)."""
    p.add_argument("--stems", nargs="+", default=None, required=required, help=help_text)


def stems_from_args(values: list[str] | None) -> list[str] | None:
    """The stems given to ``--stems`` (``None`` if it wasn't given)."""
    if values is None:
        return None
    return [stem.strip() for value in values for stem in value.split(",") if stem.strip()]


def parse_sharded_args(p: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse the arguments, check the shard index and set up logging."""
    args = p.parse_args()
    check_shard_arguments(p, args)
    setup_logging()
    return args
