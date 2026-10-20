"""Static sharding: which clips a worker of a sharded run processes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def shard_items(items: Sequence[T], shard: int, num_shards: int) -> list[T]:
    """Every ``num_shards``-th item, starting at ``shard`` (all of them for
    one shard). SA-V Stage 1 shards by a hash of the video id instead
    (:func:`~trajloc.preprocess.stage1.sav.extract.shard_video_ids`)."""
    if num_shards <= 1:
        return list(items)
    return [item for i, item in enumerate(items) if i % num_shards == shard]
