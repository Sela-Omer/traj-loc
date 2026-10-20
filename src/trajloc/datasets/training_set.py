"""The training dataset of a config."""

from __future__ import annotations

from trajloc.core.config import TrajLocConfig
from trajloc.datasets.cached_clips import CachedClipDataset
from trajloc.datasets.layout import BackboneName


def build_training_dataset(
    config: TrajLocConfig, backbone: BackboneName, split: str = "train"
) -> CachedClipDataset:
    """The config's training set, read from ``backbone``'s caches."""
    return CachedClipDataset(
        config.training_set.root,
        backbone,
        dataset_name=config.training_set.name,
        split=split,
        num_frames=config.num_frames,
        video_size=config.video_size,
        box_mask=config.box_mask,
    )
