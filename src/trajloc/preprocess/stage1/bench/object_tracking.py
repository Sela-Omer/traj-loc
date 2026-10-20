"""Benchmark Stage 1A: track every object of each clip.

Per clip (the first 49 source frames, center-cropped to 480x720 (HxW)):

1. **Foreground objects**, the benchmark's own annotation
   (:class:`~.benchmarks.Benchmark`):

   - MoveBench: frame-0 masks, propagated with SAM 2; trajectories are the
     benchmark's ground-truth tracks (visible, in-frame frames only).
   - MagicBench: per-frame instance masks; no trajectories (tracked below).

2. **Background objects**: SAM 2 automatic masks on frame 0 minus the
   foreground, propagated (:func:`~.segmentation.segment_background`).
3. **Tracking**: every object without a trajectory gets a swarm at its
   largest-mask frame; one CoTracker3 pass; the best-anchored point per
   object is its trajectory (:func:`~.clip_geometry.object_trajectory`).
   Objects without one are dropped.
4. **Records**: per surviving object (foreground first, then background),
   normalized positions and boxes, depths 0, and a highlight crop at its
   largest-mask frame (``object_crops/<stem>/<index>.png``) for naming.
5. **Write** ``videos/<stem>.mp4``, ``first_frames/<stem>.png`` and
   ``objects/<stem>.json``. Naming and prompts are Stage 1B
   (:mod:`.object_naming`), a separate step.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from trajloc.datasets.layout import (
    CLIP_HEIGHT,
    CLIP_NUM_FRAMES,
    CLIP_WIDTH,
    ERRORS_DIR,
    FIRST_FRAMES_DIR,
    OBJECT_CROPS_DIR,
    OBJECTS_DIR,
    TIMINGS_DIR,
    TMP_DIR,
    VIDEOS_DIR,
)
from trajloc.preprocess.stage1.bench.benchmarks import Benchmark, ClipSkipped
from trajloc.preprocess.stage1.bench.clip_geometry import (
    clip_source_frames,
    highlight_crop,
    object_trajectory,
    seed_swarm,
    write_sam2_frames,
)
from trajloc.preprocess.stage1.bench.segmentation import Sam2Models, segment_background
from trajloc.preprocess.stage1.geometry import CropWindow, center_crop_window
from trajloc.preprocess.stage1.records import (
    normalized_boxes,
    normalized_positions,
    write_clip_media,
    write_json,
    zero_depths,
)
from trajloc.preprocess.stage1.tracking import TRACK_LONG_SIDE, CoTracker, Trajectory

logger = logging.getLogger(__name__)

OUT_DIRS = (
    VIDEOS_DIR,
    FIRST_FRAMES_DIR,
    OBJECT_CROPS_DIR,
    OBJECTS_DIR,
    ERRORS_DIR,
    TIMINGS_DIR,
    TMP_DIR,
)


def _track_objects(
    masks: np.ndarray,
    to_track: list[int],
    source_frames: list[np.ndarray],
    picks: list[int],
    crop: CropWindow,
    H_source: int,
    W_source: int,
    cotracker: CoTracker,
) -> dict[int, Trajectory | None]:
    """The trajectory of each object of ``to_track``, from one CoTracker3
    pass over their swarms (``None``: no swarm, or no anchored point; no
    entries if nothing could be tracked)."""
    queries: list[tuple[int, np.ndarray]] = []
    query_idx: dict[int, int] = {}
    for i in to_track:
        swarm = seed_swarm(masks[i], crop, H_source, W_source)
        if swarm is None:
            continue
        best_t_idx, pts = swarm
        query_idx[i] = len(queries)
        queries.append((picks[best_t_idx], pts))
    if not queries:
        return {}
    tracked = cotracker.run(source_frames, queries)
    if tracked is None:
        return {}
    return {
        i: object_trajectory(
            masks[i], query_idx.get(i, -1), tracked.ranges, tracked.tracks, picks, crop
        )
        for i in to_track
    }


def _largest_mask_frame(masks_t: np.ndarray) -> int | None:
    """The first frame where the object's mask is largest (``None`` if always empty)."""
    best_t, best_area = None, 0
    for t in range(CLIP_NUM_FRAMES):
        area = int(masks_t[t].sum())
        if area > best_area:
            best_area, best_t = area, t
    return best_t


def _object_record(
    new_idx: int,
    instance: dict[str, Any],
    masks_t: np.ndarray,
    traj: Trajectory,
    clip_rgb_frames: list[np.ndarray],
    crops_dir: Path,
    stem: str,
) -> dict[str, Any]:
    """One object of ``objects/<stem>.json``, saving its highlight crop (at
    its largest-mask frame) to ``crops_dir/<index>.png``. Stage 1B names it."""
    rx, ry, valid = traj
    positions = normalized_positions(rx, ry, valid)
    best_t = _largest_mask_frame(masks_t)
    crop_rel = None
    if best_t is not None:
        crop_image = highlight_crop(clip_rgb_frames[best_t], masks_t[best_t])
        if crop_image is not None:
            crop_name = f"{new_idx}.png"
            crop_image.save(crops_dir / crop_name)
            crop_rel = f"{stem}/{crop_name}"
    return {
        "object_index": new_idx,
        "source_object_id": int(instance["source_object_id"]),
        "channel": instance["channel"],
        "positions": positions,
        "boxes": normalized_boxes(masks_t),
        "depths": zero_depths(positions),
        "crop": crop_rel,
        "noun": None,
        "phrase": None,
        "is_animate": None,
    }


def process_clip(
    stem: str,
    benchmark: Benchmark,
    out_dirs: dict[str, Path],
    sam2: Sam2Models,
    cotracker: CoTracker,
) -> dict:
    """Stage 1A for one clip: writes its records and returns a summary.
    Raises :class:`~.benchmarks.ClipSkipped` if the clip can't be built."""
    t0 = time.perf_counter()

    source_frames, H_source, W_source = benchmark.source.read_video_frames(stem)
    if not source_frames:
        raise ClipSkipped("no_video_frames")
    annotation = benchmark.read_annotation(stem, H_source, W_source)
    crop = center_crop_window(W_source, H_source)
    picks = clip_source_frames(len(source_frames))
    frames_dir = out_dirs[TMP_DIR] / stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    write_sam2_frames(source_frames, picks, crop, frames_dir)
    foreground_masks = benchmark.clip_masks(annotation, crop, frames_dir, sam2)
    fg_trajectories = benchmark.trajectories(annotation, picks, crop)
    n_fg = foreground_masks.shape[0]

    background_masks, stats = segment_background(sam2, frames_dir, foreground_masks)
    n_bg = background_masks.shape[0]
    logger.info(
        f"[{stem}] FG={n_fg}  BG={n_bg}  (amg_raw={stats['amg_raw']} "
        f"after_fg={stats['after_fg_drop']} seed={stats['seed_kept']})"
    )
    if n_fg + n_bg == 0:
        raise ClipSkipped("no_instances")

    instances = [{"source_object_id": j, "channel": "foreground"} for j in range(n_fg)]
    instances += [{"source_object_id": n_fg + j, "channel": "background"} for j in range(n_bg)]
    object_masks = np.concatenate([foreground_masks, background_masks], axis=0)
    n_total = len(instances)

    # Trajectories: the foreground's own, if it has them; everything else
    # (all objects, for MagicBench) from one CoTracker pass.
    object_trajectories: list[Trajectory | None] = [None] * n_total
    if fg_trajectories is not None:
        object_trajectories[:n_fg] = fg_trajectories
        to_track = list(range(n_fg, n_total))
    else:
        to_track = list(range(n_total))
    tracked = _track_objects(
        object_masks, to_track, source_frames, picks, crop, H_source, W_source, cotracker
    )
    for i, trajectory in tracked.items():
        object_trajectories[i] = trajectory

    kept = [(i, t) for i, t in enumerate(object_trajectories) if t is not None]
    if not kept:
        raise ClipSkipped("no_objects_after_tracking")

    clip_bgr_frames = [crop.warp(source_frames[t_src]) for t_src in picks]
    clip_rgb_frames = [frame[..., ::-1] for frame in clip_bgr_frames]
    crops_dir = out_dirs[OBJECT_CROPS_DIR] / stem
    crops_dir.mkdir(parents=True, exist_ok=True)
    kept_records = [
        _object_record(
            new_idx,
            instances[i],
            object_masks[i],
            trajectory,
            clip_rgb_frames,
            crops_dir,
            stem,
        )
        for new_idx, (i, trajectory) in enumerate(kept)
    ]

    write_clip_media(
        clip_bgr_frames,
        out_dirs[VIDEOS_DIR] / f"{stem}.mp4",
        out_dirs[FIRST_FRAMES_DIR] / f"{stem}.png",
    )
    write_json(
        out_dirs[OBJECTS_DIR] / f"{stem}.json",
        {
            **benchmark.objects_record_header(stem),
            "num_foreground": n_fg,
            "num_background": n_bg,
            "num_kept": len(kept_records),
            "objects": kept_records,
        },
    )

    return {
        "stem": stem,
        "num_foreground": n_fg,
        "num_background": n_bg,
        "num_kept": len(kept_records),
        "num_foreground_kept": sum(1 for i, _ in kept if i < n_fg),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }


def run(
    stems: list[str], out_dir: Path, benchmark: Benchmark, device: str = "cuda:0"
) -> tuple[int, int]:
    """Stage 1A for ``stems`` (clips with an ``objects/`` record are skipped).
    Failures are recorded in ``errors/<stem>.err``; the run goes on. Returns
    ``(done, failed)``: the clips tracked (now or before) and those that raised
    (a clip skipped for keeping no object is neither)."""
    done = failed = 0
    out_root = Path(out_dir)
    out_dirs = {name: out_root / name for name in OUT_DIRS}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"loading SAM2 (everything + video) on {device}...")
    sam2 = Sam2Models(device)
    sam2.ensure(CLIP_HEIGHT, CLIP_WIDTH)
    logger.info(f"loading CoTracker3 on {device}...")
    cotracker = CoTracker(device=device, max_resize_long_side=TRACK_LONG_SIDE)

    for stem in stems:
        if (out_dirs[OBJECTS_DIR] / f"{stem}.json").exists():
            logger.info(f"[{stem}] already tracked, skipping")
            done += 1
            continue
        try:
            result = process_clip(stem, benchmark, out_dirs, sam2, cotracker)
        except ClipSkipped as e:
            write_json(
                out_dirs[ERRORS_DIR] / f"{stem}.err",
                {"stem": stem, "error": str(e), "step": "tracking"},
            )
            logger.error(f"[{stem}] ERROR: {e}")
            continue
        except Exception as e:  # recorded per clip; the run goes on
            msg = f"{type(e).__name__}:{e}"
            error = {"stem": stem, "error": msg, "step": "tracking"}
            write_json(out_dirs[ERRORS_DIR] / f"{stem}.err", error)
            logger.error(f"[{stem}] EXCEPTION: {msg}")
            failed += 1
            continue
        finally:  # the clip's temporary frames for SAM 2
            shutil.rmtree(out_dirs[TMP_DIR] / stem, ignore_errors=True)
        (out_dirs[ERRORS_DIR] / f"{stem}.err").unlink(missing_ok=True)  # from an earlier attempt
        write_json(
            out_dirs[TIMINGS_DIR] / f"{stem}.json",
            {"stem": stem, "wall_seconds": result["wall_seconds"]},
        )
        logger.info(
            f"[{stem}] OK in {result['wall_seconds']}s  "
            f"FG={result['num_foreground']}(kept {result['num_foreground_kept']}) "
            f"BG={result['num_background']} kept={result['num_kept']}"
        )
        done += 1
    return done, failed
