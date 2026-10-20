"""Which of a clip's SVG2 objects to track.

An object is a candidate unless it was merged into a parent
(:mod:`.masks`), its label is atmospheric or abstract
(``NEVER_TRACKED_LABELS``: sky, shadow, "object", ...), or it fails the
shared size and visibility filters (:mod:`..object_filters`).

Candidates are taken largest peak area first, skipping one whose head noun
(the label's last word) already has ``PER_NOUN_CAP`` selected objects or
that duplicates a selected object with the same head noun (mask IoU at
least ``DEDUP_IOU`` on some frame), until ``top_n`` are selected.
"""

from __future__ import annotations

from dataclasses import dataclass

from trajloc.preprocess.stage1.object_filters import (
    FilterFailure,
    failed_filter,
)
from trajloc.preprocess.stage1.sav.masks import peak_mask_area

PER_NOUN_CAP = 4  # at most this many selected objects per head noun, per clip
DEDUP_IOU = 0.50  # same head noun and mask IoU >= this on some frame: a duplicate


# Atmospheric, abstract, generic-scene labels that should never be tracked.
NEVER_TRACKED_LABELS = frozenset(
    {
        "sky",
        "cloud",
        "clouds",
        "sun",
        "moon",
        "shadow",
        "reflection",
        "light",
        "lighting",
        "haze",
        "fog",
        "smoke",
        "dust",
        "mist",
        "abstract shape",
        "shape",
        "pattern",
        "stuff",
        "thing",
        "area",
        "surface",
        "edge",
        "border",
        "background",
        "scene",
        "side",
        "corner",
        # Generic 'object' hits commonly correspond to fragmentary masks
        "object",
    }
)


@dataclass
class SelectionResult:
    """The selected objects (SVG2 indices, largest first), every rejected
    object with its reason, and the selected objects' peak areas."""

    selected_indices: list[int]
    rejected: list[tuple[int, str]]
    peak_pixels: dict[int, int]


def select_objects(
    labels: list[str | None],
    masks_by_object: list[list],
    *,
    peak_areas: list[int] | None = None,
    merged: set[int] | None = None,
    H: int | None = None,
    W: int | None = None,
    top_n: int = 15,
) -> SelectionResult:
    """Select up to ``top_n`` of a clip's objects (see the module docstring).

    Args:
        labels: SVG2 label per object (``None`` if it has none).
        masks_by_object: per object, per frame, a 0/1 mask or ``None``.
        peak_areas: per object, its peak mask area (computed if ``None``).
        merged: objects merged into a parent.
        H, W: the frame size, for the huge-area cap (no cap if ``None``).
    """
    num_objects = len(labels)
    rejected: list[tuple[int, str]] = []
    merged = merged or set()
    if peak_areas is None:
        peak_areas = [peak_mask_area(masks_by_object[i]) for i in range(num_objects)]

    candidates = _candidates(labels, masks_by_object, peak_areas, merged, H, W, rejected)
    selected = _deduplicated(candidates, labels, masks_by_object, top_n, rejected)

    overflow = [
        i for _, i in candidates if i not in selected and not any(idx == i for idx, _ in rejected)
    ]
    for i in overflow:
        rejected.append((i, "over_top_n"))

    return SelectionResult(
        selected_indices=selected,
        rejected=rejected,
        peak_pixels={i: peak_areas[i] for i in selected},
    )


def _candidates(
    labels: list[str | None],
    masks_by_object: list[list],
    peak_areas: list[int],
    merged: set[int],
    H: int | None,
    W: int | None,
    rejected: list[tuple[int, str]],
) -> list[tuple[int, int]]:
    """``(peak area, index)`` of the objects passing the label, size and
    visibility filters, largest first; the others are added to ``rejected``."""
    n_visible = [
        sum(1 for m in masks_by_object[i] if m is not None and int(m.sum()) > 0)
        for i in range(len(labels))
    ]

    candidates = []
    for i, lbl in enumerate(labels):
        if i in merged:
            rejected.append((i, "merged_into_parent"))
            continue
        norm = (lbl or "").strip().lower()
        if norm in NEVER_TRACKED_LABELS:
            rejected.append((i, "atmospheric_or_abstract"))
            continue
        peak = peak_areas[i]
        failure = failed_filter(peak, n_visible[i], H, W)
        if failure is FilterFailure.TINY:
            rejected.append((i, f"tiny_area={peak}px"))
        elif failure is FilterFailure.HUGE:
            rejected.append((i, f"huge_area={peak}px"))
        elif failure is FilterFailure.LOW_VISIBILITY:
            rejected.append((i, f"low_visibility={n_visible[i]}"))
        else:
            candidates.append((peak, i))
    candidates.sort(reverse=True)  # largest first
    return candidates


def _head_noun(label: str | None) -> str:
    """The label's last word, lowercase ('glass window' -> 'window')."""
    lbl = (label or "").lower()
    return lbl.split()[-1] if lbl else ""


def _max_iou(masks_a: list, masks_b: list) -> float:
    """The largest mask IoU over the frames where both have a mask (stops
    early once it reaches ``DEDUP_IOU``)."""
    best = 0.0
    for t in range(len(masks_a)):
        mi = masks_a[t]
        mj = masks_b[t]
        if mi is None or mj is None:
            continue
        a = mi.astype(bool)
        b = mj.astype(bool)
        inter = (a & b).sum()
        union = (a | b).sum()
        if union > 0:
            iou = inter / union
            if iou > best:
                best = iou
                if best >= DEDUP_IOU:
                    break
        if best >= DEDUP_IOU:
            break
    return best


def _deduplicated(
    candidates: list[tuple[int, int]],
    labels: list[str | None],
    masks_by_object: list[list],
    top_n: int,
    rejected: list[tuple[int, str]],
) -> list[int]:
    """The candidates in order, skipping any whose head noun already has
    ``PER_NOUN_CAP`` selected objects or that duplicates a selected object
    with the same head noun, until ``top_n`` are selected. SVG2 sometimes
    segments one physical object as several masks, or stacks near-identical
    instances (13 umbrellas)."""
    selected: list[int] = []
    per_noun_count: dict[str, int] = {}
    for _peak, i in candidates:
        head = _head_noun(labels[i])
        if per_noun_count.get(head, 0) >= PER_NOUN_CAP:
            rejected.append((i, f"per_noun_cap={PER_NOUN_CAP}"))
            continue
        duplicate_of = None
        for j in selected:
            if _head_noun(labels[j]) != head:
                continue
            best = _max_iou(masks_by_object[i], masks_by_object[j])
            if best >= DEDUP_IOU:
                rejected.append((i, f"dup_iou={best:.2f}_with_{j}"))
                duplicate_of = j
                break
        if duplicate_of is not None:
            continue
        selected.append(i)
        per_noun_count[head] = per_noun_count.get(head, 0) + 1
        if len(selected) >= top_n:
            break
    return selected
