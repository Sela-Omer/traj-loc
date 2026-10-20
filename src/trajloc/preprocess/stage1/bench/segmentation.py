"""Object masks for the benchmarks with SAM 2.1 (hiera-large).

- **Background objects** (:func:`segment_background`): SAM 2's automatic
  mask generator on frame 0 (coarse settings: whole objects, not parts).
  Candidates that are small, mostly inside the foreground, or nested in a
  larger candidate are dropped, and the ``MAX_BG_OBJECTS`` (4) largest of
  the rest are propagated through the clip with the SAM 2 video predictor,
  minus the foreground, then filtered by shape and by the SA-V selection
  thresholds (size, visibility). So at most 4 are kept, possibly fewer.
- **Foreground propagation** (:func:`propagate_masks`, MoveBench only): the
  benchmark's frame-0 masks, propagated through the clip.

Checkpoint: ``$TRAJLOC_SAM2_CHECKPOINT``, default
``<project>/checkpoints/sam2.1_hiera_large.pt``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from trajloc.datasets.layout import CLIP_NUM_FRAMES
from trajloc.paths import project_root
from trajloc.preprocess.stage1.object_filters import (
    MIN_PEAK_AREA_PX,
    FilterFailure,
    failed_filter,
)

SAM2_CKPT = Path(
    os.environ.get("TRAJLOC_SAM2_CHECKPOINT", project_root() / "checkpoints/sam2.1_hiera_large.pt")
)

SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

MAX_FOREGROUND_OVERLAP = 0.80  # >= this frac of a candidate inside FG -> drop

CONTAINMENT_FRAC = 0.85  # >= this frac of a mask inside a larger kept mask -> drop (nested dedup)

MAX_BOX_ASPECT_RATIO = 8.0  # bounding-box aspect ratio above this -> a sliver, drop

# At most the 4 largest background candidates (the same area-ranked top-N rule
# as SA-V's selection): the dominant scene context, without clutter. A larger
# cap over-segmented uniform scenes (duplicate water and sky patches).
MAX_BG_OBJECTS = 4

# SAM 2 automatic mask generator settings. Coarse, for whole objects rather
# than parts: few points and strong NMS keep the candidates few;
# min_mask_region_area drops specks.
AMG_POINTS_PER_SIDE = 16

AMG_PRED_IOU_THRESHOLD = 0.80

AMG_STABILITY_SCORE_THRESHOLD = 0.90

AMG_BOX_NMS_THRESHOLD = 0.50

AMG_MIN_REGION_FRAC = 0.0015  # * H*W -> min_mask_region_area (drop tiny specks)


@dataclass
class Sam2Models:
    """SAM 2.1 on one device: the automatic mask generator (on the image
    model) and the video predictor, built on the first :meth:`ensure`."""

    device: str = "cuda:0"
    mask_generator: Any = None
    video_predictor: Any = None
    image_model: Any = None

    def ensure(self, H: int, W: int) -> None:
        """Build the models (once). ``H x W`` sets the generator's minimum
        region area, so the first call fixes it for the run."""
        if self.video_predictor is not None:
            return
        if not SAM2_CKPT.is_file():
            raise FileNotFoundError(
                f"no SAM 2.1 checkpoint at {SAM2_CKPT}: scripts/prepare_data.sh downloads it "
                f"(or set TRAJLOC_SAM2_CHECKPOINT)"
            )
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2, build_sam2_video_predictor

        self.image_model = build_sam2(
            SAM2_CFG, str(SAM2_CKPT), device=self.device, apply_postprocessing=False
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.image_model,
            points_per_side=AMG_POINTS_PER_SIDE,
            pred_iou_thresh=AMG_PRED_IOU_THRESHOLD,
            stability_score_thresh=AMG_STABILITY_SCORE_THRESHOLD,
            box_nms_thresh=AMG_BOX_NMS_THRESHOLD,
            min_mask_region_area=int(AMG_MIN_REGION_FRAC * H * W),
            output_mode="binary_mask",
        )
        self.video_predictor = build_sam2_video_predictor(
            SAM2_CFG, str(SAM2_CKPT), device=self.device
        )


def _containment_frac(small: np.ndarray, big: np.ndarray) -> float:
    s = small.sum()
    if s == 0:
        return 0.0
    return float(np.logical_and(small, big).sum()) / float(s)


def _dedup_nested(masks: list[np.ndarray]) -> list[int]:
    """Keep largest-first; drop a mask that is >= CONTAINMENT_FRAC inside an
    already-kept (larger) mask. Returns kept indices into `masks`."""
    order = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))
    kept: list[int] = []
    for i in order:
        if any(_containment_frac(masks[i], masks[k]) >= CONTAINMENT_FRAC for k in kept):
            continue
        kept.append(i)
    return kept


def propagate_masks(
    predictor, frames_dir: Path, frame0_masks: list[np.ndarray], empty_hw: tuple[int, int]
) -> np.ndarray:
    """Propagate frame-0 masks through the clip's frames in ``frames_dir``
    with the SAM 2 video predictor.

    Returns ``(n, CLIP_NUM_FRAMES, H, W)`` bool (``H, W`` of the frames), all
    ``False`` where an object was not propagated. ``empty_hw`` is the
    ``H, W`` returned when nothing is (no masks, or no frame).
    """
    if not frame0_masks:
        return np.zeros((0, CLIP_NUM_FRAMES, *empty_hw), dtype=np.bool_)
    state = predictor.init_state(video_path=str(frames_dir))
    for k, m in enumerate(frame0_masks):
        predictor.add_new_mask(
            inference_state=state, frame_idx=0, obj_id=k, mask=m.astype(np.uint8)
        )
    out: dict[int, list[np.ndarray | None]] = {
        k: [None] * CLIP_NUM_FRAMES for k in range(len(frame0_masks))
    }
    frame_hw: tuple[int, int] | None = None  # the propagated masks' H, W
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        if frame_idx >= CLIP_NUM_FRAMES:
            break
        for j, oid in enumerate(obj_ids):
            mask = (mask_logits[j] > 0.0).cpu().numpy().astype(np.uint8)
            if mask.ndim == 3:
                mask = mask[0]
            out[int(oid)][frame_idx] = mask
            if frame_hw is None:
                frame_hw = (mask.shape[0], mask.shape[1])
    num_objects = len(frame0_masks)
    if frame_hw is None:
        return np.zeros((num_objects, CLIP_NUM_FRAMES, *empty_hw), dtype=np.bool_)
    stacked = np.zeros((num_objects, CLIP_NUM_FRAMES, *frame_hw), dtype=np.bool_)
    for k in range(num_objects):
        for t in range(CLIP_NUM_FRAMES):
            propagated = out[k][t]
            if propagated is not None:
                stacked[k, t] = propagated.astype(bool)
    return stacked


def filter_background_candidates(masks: np.ndarray) -> tuple[list[int], dict[int, str]]:
    """The objects of ``(n, T, H, W)`` masks that pass the shared size and
    visibility filters (:mod:`..object_filters`) and whose union's bounding
    box is at most ``MAX_BOX_ASPECT_RATIO`` : 1, and why each other one was dropped."""
    num_objects, _T, H, W = masks.shape
    kept, reasons = [], {}
    for k in range(num_objects):
        areas = masks[k].sum(axis=(1, 2))  # per frame
        peak = int(areas.max())
        n_visible = int((areas > 0).sum())
        failure = failed_filter(peak, n_visible, H, W)
        if failure is not None:
            reasons[k] = {
                FilterFailure.TINY: f"tiny_peak_px={peak}",
                FilterFailure.HUGE: f"huge_peak_frac={peak / (H * W):.2f}",
                FilterFailure.LOW_VISIBILITY: f"low_visibility={n_visible}",
            }[failure]
            continue
        ys, xs = np.where(masks[k].any(axis=0))  # the union over frames
        if ys.size == 0:
            reasons[k] = "empty_union"
            continue
        box_w = xs.max() - xs.min() + 1
        box_h = ys.max() - ys.min() + 1
        aspect_ratio = max(box_w, box_h) / max(1, min(box_w, box_h))
        if aspect_ratio > MAX_BOX_ASPECT_RATIO:
            reasons[k] = f"sliver_ar={aspect_ratio:.1f}"
            continue
        kept.append(k)
    return kept, reasons


def segment_background(
    sam2: Sam2Models, frames_dir: Path, fg_per_frame: np.ndarray
) -> tuple[np.ndarray, dict]:
    """The clip's background objects (see the module docstring).

    Args:
        sam2: the SAM 2 models.
        frames_dir: the clip's frames, ``00000.jpg`` onwards.
        fg_per_frame: ``(n_fg, T, H, W)`` bool foreground masks.

    Returns:
        ``(n_bg, T, H, W)`` bool masks, and counts of the candidates after
        each filter (``amg_raw``, ``after_fg_drop``, ``seed_kept``,
        ``pre_shape``, ``kept``) with the shape filter's
        ``rejection_reasons``.
    """
    fg_union = fg_per_frame.any(axis=0)  # (T, H, W)
    image_rgb = np.array(Image.open(frames_dir / "00000.jpg").convert("RGB"))
    H, W = image_rgb.shape[:2]
    sam2.ensure(H, W)

    # 1) Automatic masks on frame 0.
    with torch.inference_mode():
        anns = sam2.mask_generator.generate(image_rgb)
    n_raw = len(anns)

    # 2) Drop FG-overlapping candidates (small intersections survive).
    fg0 = fg_union[0]
    cand = []
    for a in anns:
        m = a["segmentation"].astype(bool)
        area = int(m.sum())
        if area < MIN_PEAK_AREA_PX:
            continue
        inside_fg = np.logical_and(m, fg0).sum() / max(1, area)
        if inside_fg >= MAX_FOREGROUND_OVERLAP:
            continue
        cand.append((m, area))
    n_after_fg = len(cand)

    # 3) Nested/duplicate dedup (largest-first containment).
    cand_masks = [m for m, _ in cand]
    keep_idx = _dedup_nested(cand_masks)
    # Largest-first, capped.
    keep_idx = sorted(keep_idx, key=lambda i: -cand[i][1])[:MAX_BG_OBJECTS]
    seed_masks = [cand_masks[i] for i in keep_idx]
    n_seed = len(seed_masks)

    def stats(pre_shape: int, kept: int, reasons: dict[int, str]) -> dict[str, Any]:
        return {
            "amg_raw": n_raw,
            "after_fg_drop": n_after_fg,
            "seed_kept": n_seed,
            "pre_shape": pre_shape,
            "kept": kept,
            "rejection_reasons": {str(k): v for k, v in reasons.items()},
        }

    if not seed_masks:
        return np.zeros((0, CLIP_NUM_FRAMES, H, W), dtype=bool), stats(0, 0, {})

    # 4) Propagate (all-empty masks of the frame size if SAM 2 propagates
    # nothing) + per-frame FG-subtract.
    bg_stack = propagate_masks(sam2.video_predictor, frames_dir, seed_masks, empty_hw=(H, W))
    bg_stack = np.logical_and(bg_stack, ~fg_union[None, :, :, :])

    # 5) Shape filter.
    kept, reasons = filter_background_candidates(bg_stack)
    return bg_stack[kept], stats(int(bg_stack.shape[0]), len(kept), reasons)
