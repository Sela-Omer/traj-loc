"""Datasets over preprocessed (Stage-2) dataset roots.

- :func:`build_training_dataset` / :func:`cached_collate`: the training set a
  config describes, and its batch collation.
- :func:`enumerate_eval_samples`: the clips an inference run generates.
- :mod:`trajloc.datasets.eval_sets`: the evaluation sets.
- :mod:`trajloc.datasets.layout`: the on-disk layout every stage shares.
"""

from trajloc.datasets.cached_clips import CachedClipDataset, ClipSample
from trajloc.datasets.caches import LatentCache, TextCache
from trajloc.datasets.collate import cached_collate
from trajloc.datasets.eval_samples import EvalSample, enumerate_eval_samples
from trajloc.datasets.eval_sets import EVAL_SETS, EvalSet, ResolvedEvalSet, resolve_eval_sets
from trajloc.datasets.layout import BACKBONE_NAMES, BackboneName, DatasetRoot
from trajloc.datasets.training_set import build_training_dataset

__all__ = [
    "BACKBONE_NAMES",
    "EVAL_SETS",
    "BackboneName",
    "CachedClipDataset",
    "ClipSample",
    "DatasetRoot",
    "EvalSample",
    "EvalSet",
    "LatentCache",
    "ResolvedEvalSet",
    "TextCache",
    "build_training_dataset",
    "cached_collate",
    "enumerate_eval_samples",
    "resolve_eval_sets",
]
