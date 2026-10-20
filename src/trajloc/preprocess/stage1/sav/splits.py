"""Scene-disjoint train/test split for SA-V.

Stage 1 cuts several clips (``sav_NNNNNN_cKK``) out of each SA-V source
video (a *scene*, ``sav_NNNNNN``). Splitting clips independently would put
the same scene in both train and test, so the split is made per scene:
every clip of a held-out scene goes to test.

Scenes are ranked by the MD5 of their id, so the split is deterministic.
Two ways to choose the held-out scenes:

- ``test_scenes=N``: the N scenes with the smallest hash, an exact count.
  The released SA-V split uses ``N = 100``. Adding scenes can change it.
- ``test_pct=P``: every scene with ``hash % 10000 < round(P * 100)``, a
  per-scene rule, so adding scenes never moves an existing one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

STEM_RE = re.compile(r"^(sav_\d+)_c\d+$")


def scene_id(stem: str) -> str:
    """``"sav_000123_c04"`` -> ``"sav_000123"``."""
    m = STEM_RE.match(stem)
    if m is None:
        raise ValueError(f"stem {stem!r} doesn't match SA-V scene_clip pattern")
    return m.group(1)


def scene_hash(scene: str) -> int:
    """First 32 bits of the scene id's MD5. Shared by both selection modes,
    so a scene held out at a small N stays held out at any larger N."""
    return int(hashlib.md5(scene.encode()).hexdigest()[:8], 16)


def is_test_pct(scene: str, test_pct: float) -> bool:
    """Percent rule, quantized to basis points (1/10000)."""
    return (scene_hash(scene) % 10000) < round(test_pct * 100)


def select_test_scenes(scenes: list[str], *, test_pct: float, test_scenes: int | None) -> set[str]:
    """Held-out scenes: the ``test_scenes`` smallest hashes if given, else
    the percent rule."""
    if test_scenes is not None:
        if test_scenes < 0 or test_scenes > len(scenes):
            raise ValueError(f"--test_scenes {test_scenes} out of range [0, {len(scenes)}]")
        return set(sorted(scenes, key=scene_hash)[:test_scenes])
    return {s for s in scenes if is_test_pct(s, test_pct)}


def build_splits(
    root: Path, out: Path, *, test_pct: float = 5.0, test_scenes: int | None = None
) -> dict[str, list[str]]:
    """Split the clips in ``root/videos`` and write
    ``{"train": [...], "test": [...]}`` (sorted stems) to ``out``."""
    stems = sorted(p.stem for p in (root / "videos").glob("*.mp4"))
    logger.info(f"{len(stems)} clips under {root / 'videos'}")

    scene_to_stems: dict[str, list[str]] = {}
    for s in stems:
        scene_to_stems.setdefault(scene_id(s), []).append(s)
    logger.info(f"{len(scene_to_stems)} unique scenes")

    test_set = select_test_scenes(list(scene_to_stems), test_pct=test_pct, test_scenes=test_scenes)
    if test_scenes is not None:
        logger.info(f"selection: the {test_scenes} smallest-hash scenes")
    else:
        logger.info(f"selection: hash % 10000 < {round(test_pct * 100)}")

    train: list[str] = []
    test: list[str] = []
    for scene, scene_stems in sorted(scene_to_stems.items()):
        (test if scene in test_set else train).extend(scene_stems)
    train.sort()
    test.sort()

    splits = {"train": train, "test": test}
    with open(out, "w") as f:
        json.dump(splits, f)
    logger.info(
        f"wrote {out}: train={len(train)} clips "
        f"({len(scene_to_stems) - len(test_set)} scenes), "
        f"test={len(test)} clips ({len(test_set)} scenes)"
    )

    overlap = {scene_id(s) for s in train} & {scene_id(s) for s in test}
    if overlap:
        raise RuntimeError(
            f"scene leakage: {len(overlap)} scenes in both splits, e.g. {sorted(overlap)[:3]}"
        )
    return splits
