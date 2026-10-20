"""Compositional mask merging: fuse 'part' masks into their parent.

When SVG2 labels a person plus their separate jacket/coat/trousers, those
clothing masks share image region with the person's mask (the person mask
is often torso-only and the jacket extends it). Treating them as separate
trajectories produces 4 entangled <traj_i> tokens for one body, which is
both redundant and noisy for trajectory training.

A part (label in ``PART_LABELS``) is merged into a parent (label in
``PARENT_LABELS``, non-empty mask) if, on some frame where both are present:

- (a) at least ``COVERAGE_THRESHOLD`` of the part's mask is inside the parent's;
- (b) their centers of mass are within ``DISTANCE_THRESHOLD`` of the image
  diagonal and their boxes overlap (IoU above ``TOUCH_IOU``); or
- (c) at least ``BOX_CONTAIN_THRESHOLD`` of the part's box is inside the
  parent's (SVG2 sometimes splits a person and their clothing into adjacent,
  non-overlapping masks).

Each measure is its maximum over the frames. Of the qualifying parents, the
part goes to the one with the highest score (coverage, plus
``TOUCH_BONUS`` for (b) and ``CONTAIN_BONUS`` for (c)): the part's masks are
unioned into the parent's on every frame and the part is dropped. The
parent's statistics are refreshed, so later parts see the grown parent.

Pairs this covers: clothing and body parts -> a person or animal; vehicle
parts (wheel, headlight, hood, ...) -> the vehicle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Labels that act as PARENTS for compositional merging.
PARENT_LABELS = frozenset(
    {
        # Animate
        "person",
        "man",
        "woman",
        "child",
        "boy",
        "girl",
        "kid",
        "baby",
        "dancer",
        "judoka",
        "cyclist",
        "biker",
        "athlete",
        "runner",
        "shopper",
        "pedestrian",
        "soldier",
        "monk",
        "tourist",
        # Vehicles
        "car",
        "sedan",
        "suv",
        "truck",
        "van",
        "minivan",
        "bus",
        "lorry",
        "motorcycle",
        "bicycle",
        "scooter",
        "moped",
        "auto rickshaw",
        "rickshaw",
        "tractor",
        "police car",
        "taxi",
        "ambulance",
        "boat",
        "ship",
        "yacht",
        "kayak",
        "canoe",
        "airplane",
        "helicopter",
        "drone",
        "horse",
        "cow",
        "sheep",
        "dog",
        "cat",  # animals can have parts too
    }
)

TOUCH_IOU = 0.05  # (b): box IoU above this, with close centers, is touching
TOUCH_BONUS = 0.3  # score bonus for (b)
CONTAIN_BONUS = 0.2  # score bonus for (c)

# Labels of parts, merged into a parent that hosts them.
PART_LABELS = frozenset(
    {
        # Body parts (for animate parents)
        "arm",
        "hand",
        "hands",
        "leg",
        "foot",
        "feet",
        "head",
        "shoulder",
        "knee",
        "finger",
        "toe",
        "wrist",
        "elbow",
        "ankle",
        "hip",
        "neck",
        "back",
        "chest",
        "human arm",
        "human hand",
        "thumb",
        "ear",
        "eye",
        "nose",
        "mouth",
        "tongue",
        "lip",
        "lips",
        "chin",
        "forehead",
        "hair",
        "braid",
        "beard",
        "skin",
        "tooth",
        "teeth",
        "fingertip",
        "palm",
        "knuckle",
        "eyebrow",
        "eyelash",
        "human leg",
        "human foot",
        "human head",
        "human face",
        "face",
        "fingertips",
        "pinky",
        "ring finger",
        "tail",
        # Clothing
        "trousers",
        "pants",
        "jeans",
        "skirt",
        "shirt",
        "t-shirt",
        "shorts",
        "dress",
        "coat",
        "jacket",
        "sweater",
        "sweatshirt",
        "hoodie",
        "blouse",
        "scarf",
        "tie",
        "belt",
        "shoe",
        "shoes",
        "sock",
        "socks",
        "boot",
        "boots",
        "sandal",
        "sandals",
        "sneaker",
        "sneakers",
        "hat",
        "cap",
        "helmet",
        "glove",
        "gloves",
        "headscarf",
        "headband",
        "uniform",
        "vest",
        "fabric",
        "sleeve",
        "collar",
        "pocket",
        "button",
        "headwear",
        "footwear",
        "leggings",
        "swimsuit",
        "tank top",
        "polo shirt",
        "garment",
        "clothing",
        "apparel",
        "outfit",
        "robe",
        "puffer jacket",
        "suit jacket",
        "cycling jersey",
        # Vehicle parts
        "wheel",
        "wheels",
        "tire",
        "tires",
        "rim",
        "headlight",
        "taillight",
        "windshield",
        "rear window",
        "side window",
        "windowpane",
        "bumper",
        "fender",
        "hood",
        "trunk",
        "spoiler",
        "license plate",
        "side mirror",
        "rearview mirror",
        "antenna",
        "door handle",
        "fuel tank",
        "gas cap",
        "exhaust pipe",
        "muffler",
        "car door",
        "car window",
        "car wheel",
        "vehicle wheel",
        "front grille",
        "grille",
        "hub cap",
        "hubcap",
        "wiper",
        "windshield wiper",
        "motorcycle seat",
        "motorcycle fuel tank",
        "handlebar",
        "handlebars",
        "saddle",
        "pedal",
        "pedals",
        "chain",
        "kickstand",
        "frame",
        "bicycle wheel",
        "bicycle frame",
        "bicycle saddle",
        "boxy cab",
        "cargo area",
        "car hood",
        "car dashboard control panel",
        "dashboard",
        "steering wheel",
        "control panel",
        "gauge",
        # Animal parts (only relevant if animal is parent)
        "fur",
        "feathers",
        "beak",
        "wing",
        "wings",
        "paw",
        "paws",
        "claw",
        "claws",
        "fin",
        "fins",
        "scale",
        "scales",
        "horn",
        "horns",
    }
)

# A part is never a parent (the two label sets are disjoint).
COVERAGE_THRESHOLD = 0.5  # (a): the part's mask this much inside the parent's
DISTANCE_THRESHOLD = 0.05  # (b): centers of mass this close, as a fraction of the diagonal
BOX_CONTAIN_THRESHOLD = 0.6  # (c): the part's box this much inside the parent's

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1), inclusive pixel bounds (see below)
Masks = list[list[np.ndarray | None]]  # per object, per frame, a 0/1 mask or None


@dataclass
class MergeResult:
    """The masks after merging (a copy: the parts' masks unioned into their
    parents'), the merged parts, each part's parent, and every object's peak
    mask area after the merge."""

    masks_by_object: Masks
    merged: set[int]
    parent_of: dict[int, int]
    peak_areas: list[int]


def peak_mask_area(masks: list[np.ndarray | None]) -> int:
    """An object's largest single-frame mask area, in pixels (0 if it has none)."""
    return max((int(m.sum()) for m in masks if m is not None), default=0)


# Box convention: ``_stats`` returns inclusive pixel boxes ``(x0, y0, x1, y1)``,
# but the box measures below take areas as ``x1 - x0`` (as if exclusive), and
# ``_parent_evidence`` compares masks when boxes touch (``<=``) while
# ``_coverage_with_stats`` needs a strict overlap. The merges depend on this
# arithmetic exactly.


def _box_intersection(a: Box, b: Box) -> Box:
    """The bounds of two boxes' intersection (empty when ``x0 > x1`` or ``y0 > y1``)."""
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def _intersection_area(a: Box, b: Box) -> int:
    ix0, iy0, ix1, iy1 = _box_intersection(a, b)
    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


def _box_area(box: Box) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def _box_iou(a: Box, b: Box) -> float:
    """IoU of two boxes (areas as in the convention above)."""
    inter = _intersection_area(a, b)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union else 0.0


def _box_containment(part: Box, parent: Box) -> float:
    """The fraction of the part's box inside the parent's (areas as above)."""
    part_area = _box_area(part)
    return _intersection_area(part, parent) / part_area if part_area else 0.0


def _coverage_with_stats(
    part_mask: np.ndarray, parent_mask: np.ndarray, part_box: Box, parent_box: Box, part_area: int
) -> float:
    """The fraction of the part's mask inside the parent's, comparing the
    masks only inside the boxes' (strict) intersection."""
    if part_area == 0:
        return 0.0
    ix0, iy0, ix1, iy1 = _box_intersection(part_box, parent_box)
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    part_slice = part_mask[iy0 : iy1 + 1, ix0 : ix1 + 1].astype(bool)
    parent_slice = parent_mask[iy0 : iy1 + 1, ix0 : ix1 + 1].astype(bool)
    return int((part_slice & parent_slice).sum()) / float(part_area)


def _stats(mask: np.ndarray | None) -> tuple[Box | None, tuple[float, float] | None, int]:
    """The mask's inclusive box, center of mass and area (``None, None, 0`` if empty)."""
    if mask is None:
        return None, None, 0
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None, 0
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    com = (float(xs.mean()), float(ys.mean()))
    return box, com, int(len(xs))


class _MaskStats:
    """Per object and frame: the mask's box, center of mass and area
    (``None, None, 0`` without a mask)."""

    def __init__(self, masks_by_object: Masks, objects: set[int], n_frames: int) -> None:
        self.boxes: dict[int, list[Box | None]] = {i: [None] * n_frames for i in objects}
        self.coms: dict[int, list[tuple[float, float] | None]] = {
            i: [None] * n_frames for i in objects
        }
        self.areas: dict[int, list[int]] = {i: [0] * n_frames for i in objects}
        for i in objects:
            for t in range(n_frames):
                self.refresh(i, t, masks_by_object[i][t])

    def refresh(self, i: int, t: int, mask: np.ndarray | None) -> None:
        self.boxes[i][t], self.coms[i][t], self.areas[i][t] = _stats(mask)


def merge_parts_into_parents(
    masks_by_object: Masks, labels: list[str | None], H: int, W: int
) -> MergeResult:
    """Merge part masks into their parents (see the module docstring); the
    input is not modified.

    Args:
        masks_by_object: per object, per frame, a 0/1 mask or ``None``.
        labels: per object, its SVG2 label (or ``None``).
        H, W: the frame size (for the touching distance).
    """
    num_objects = len(labels)
    n_frames = len(masks_by_object[0]) if num_objects else 0
    touch_dist = DISTANCE_THRESHOLD * math.sqrt(H * H + W * W)

    normalized = [(label or "").lower().strip() for label in labels]
    parts = [i for i, label in enumerate(normalized) if label in PART_LABELS]
    parent_candidates = [i for i, label in enumerate(normalized) if label in PARENT_LABELS]
    if not parts:
        return MergeResult(
            masks_by_object=[list(row) for row in masks_by_object],
            merged=set(),
            parent_of={},
            peak_areas=[peak_mask_area(masks_by_object[i]) for i in range(num_objects)],
        )

    stats = _MaskStats(masks_by_object, set(parts) | set(parent_candidates), n_frames)
    # Only parents with a non-empty mask can absorb parts.
    parents = [i for i in parent_candidates if max(stats.areas[i]) > 0]

    merged = set()
    parent_of = {}
    masks_out = [list(row) for row in masks_by_object]  # merged masks replace entries
    for part in parts:
        parent = _best_parent(part, parents, masks_out, stats, n_frames, touch_dist)
        if parent is None:
            continue
        _merge_into(part, parent, masks_out, stats, n_frames)
        merged.add(part)
        parent_of[part] = parent

    return MergeResult(
        masks_by_object=masks_out,
        merged=merged,
        parent_of=parent_of,
        peak_areas=[peak_mask_area(masks_out[i]) for i in range(num_objects)],
    )


def _best_parent(
    part: int, parents: list[int], masks: Masks, stats: _MaskStats, n_frames: int, touch_dist: float
) -> int | None:
    """The qualifying parent with the highest score (the first on ties), or
    ``None``. See the module docstring for the criteria and the score."""
    best_parent = None
    best_score = 0.0
    for parent in parents:
        coverage, touching, containment = _parent_evidence(
            part, parent, masks, stats, n_frames, touch_dist
        )
        contained = containment >= BOX_CONTAIN_THRESHOLD
        score = (
            coverage + (TOUCH_BONUS if touching else 0.0) + (CONTAIN_BONUS if contained else 0.0)
        )
        qualifies = coverage >= COVERAGE_THRESHOLD or touching or contained
        if score > best_score and qualifies:
            best_score = score
            best_parent = parent
    return best_parent


def _parent_evidence(
    part: int, parent: int, masks: Masks, stats: _MaskStats, n_frames: int, touch_dist: float
) -> tuple[float, bool, float]:
    """Over the frames where both have a mask: the part's best coverage by
    the parent, whether they ever touch, and the best box containment."""
    best_coverage = 0.0
    touching = False
    best_containment = 0.0
    for t in range(n_frames):
        part_box, parent_box = stats.boxes[part][t], stats.boxes[parent][t]
        if part_box is None or parent_box is None:
            continue
        best_containment = max(best_containment, _box_containment(part_box, parent_box))
        # Masks are compared only where the boxes intersect.
        ix0, iy0, ix1, iy1 = _box_intersection(part_box, parent_box)
        part_mask, parent_mask = masks[part][t], masks[parent][t]
        if ix0 <= ix1 and iy0 <= iy1 and part_mask is not None and parent_mask is not None:
            coverage = _coverage_with_stats(
                part_mask, parent_mask, part_box, parent_box, stats.areas[part][t]
            )
            best_coverage = max(best_coverage, coverage)
        if not touching:
            part_com, parent_com = stats.coms[part][t], stats.coms[parent][t]
            if part_com and parent_com:
                dx, dy = part_com[0] - parent_com[0], part_com[1] - parent_com[1]
                distance = math.sqrt(
                    dx * dx + dy * dy
                )  # not hypot, which can differ in the last bit
                touching = distance <= touch_dist and _box_iou(part_box, parent_box) > TOUCH_IOU
    return best_coverage, touching, best_containment


def _merge_into(part: int, parent: int, masks: Masks, stats: _MaskStats, n_frames: int) -> None:
    """Union the part's masks into the parent's on every frame, refreshing the
    parent's statistics so later parts see the grown parent."""
    for t in range(n_frames):
        part_mask, parent_mask = masks[part][t], masks[parent][t]
        if part_mask is None:
            continue
        if parent_mask is None:
            masks[parent][t] = part_mask.copy()
        else:
            masks[parent][t] = (part_mask | parent_mask).astype(np.uint8)
        stats.refresh(parent, t, masks[parent][t])
