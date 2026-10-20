"""The evaluation sets inference runs on.

Each entry names a dataset root under ``<project>/data/`` (a symlink to the
preprocessed dataset). :func:`resolve_eval_sets` turns
entries into the concrete cache paths of one backbone.

To add an evaluation set, preprocess it (Stage 1 + 2), link its root to
``data/<name>`` and append an :class:`EvalSet` to :data:`EVAL_SETS` (append
only: CogVideoX's sharded numbering, and so its seeds, walks the sets in
order).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trajloc.datasets.layout import BackboneName, DatasetRoot


@dataclass(frozen=True)
class EvalSet:
    """One evaluation set.

    Attributes:
        name: its name (the output subdirectory of an inference run).
        data_subdir: its dataset root, relative to ``<project>/data/``.
    """

    name: str
    data_subdir: str

    def resolve(self, backbone: BackboneName, project_root: str | Path) -> ResolvedEvalSet:
        """Absolute paths of this set's caches for ``backbone``."""
        root = DatasetRoot(Path(project_root) / "data" / self.data_subdir)
        return ResolvedEvalSet(
            name=self.name,
            text_cache_dir=str(root.text_cache_dir(backbone)),
            latent_dir=str(root.latent_cache_dir(backbone)),
            videos_dir=str(root.videos_dir),
        )


@dataclass(frozen=True)
class ResolvedEvalSet:
    """Absolute paths of one evaluation set for one backbone."""

    name: str
    text_cache_dir: str
    latent_dir: str
    videos_dir: str


EVAL_SETS: dict[str, EvalSet] = {
    s.name: s
    for s in (
        EvalSet(name="magicbench", data_subdir="magicbench"),
        EvalSet(name="movebench", data_subdir="movebench"),
    )
}


def resolve_eval_sets(
    backbone: BackboneName,
    project_root: str | Path,
    names: list[str] | None = None,
) -> list[ResolvedEvalSet]:
    """Resolve the named evaluation sets (default: all, in registry order)."""
    if names is None:
        return [s.resolve(backbone, project_root) for s in EVAL_SETS.values()]
    unknown = [n for n in names if n not in EVAL_SETS]
    if unknown:
        raise KeyError(f"Unknown evaluation sets {unknown}. Known: {sorted(EVAL_SETS)}")
    return [EVAL_SETS[n].resolve(backbone, project_root) for n in names]
