"""SA-V clips: SA-V videos + SVG2 scene graphs and masks -> Stage-1 clips.

Per source video (up to 3 clips of 49 frames, center-cropped to 480x720 (HxW)):

1. **Select objects** per clip from the SVG2 masks: merge part masks into
   their whole (:mod:`.masks`), then keep up to 15 objects by size, dropping
   atmospheric/abstract labels, tiny/huge or rarely visible masks and
   near-duplicates (:mod:`.selection`). Clips with fewer than 2 objects are
   dropped.
2. **Seed a swarm** per object at its largest in-crop mask frame
   (:mod:`trajloc.preprocess.stage1.swarm`).
3. **Track** every clip's swarms in one CoTracker3 pass over the video.
4. **Pick the trajectory**: per object, the swarm point that stays on the
   mask longest, valid only inside the mask's bounding box
   (:func:`~trajloc.preprocess.stage1.tracking.anchor_trajectory`). Objects
   without one are dropped (again: fewer than 2 left drops the clip).
5. **Prompt**: one sentence naming the surviving objects, from the scene
   graph (:mod:`.prompts`). Trajectories and prompt entries share the index
   ``i`` (``<traj_i>``) and record the SVG2 object index; this identity is
   checked before anything is written.
6. **Write** ``videos/<stem>.mp4``, ``first_frames/<stem>.png``,
   ``trajectories/<stem>.json`` (positions, boxes, depths all 0: SA-V has
   no depth) and ``prompts/<stem>.json``, ``stem = <video>_c<clip>``.

Steps 1-2 run per clip, step 3 once per video, steps 4-6 per clip.
"""

from __future__ import annotations

import json
import logging
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
import pyarrow.parquet as pq
from pycocotools import mask as cocomask

from trajloc.datasets.layout import (
    CLIP_DIRS,
    CLIP_HEIGHT,
    CLIP_NUM_FRAMES,
    CLIP_WIDTH,
    DONE_DIR,
    ERRORS_DIR,
    FIRST_FRAMES_DIR,
    PROMPTS_DIR,
    TIMINGS_DIR,
    TRAJECTORIES_DIR,
    VIDEOS_DIR,
)
from trajloc.preprocess.stage1.geometry import CropWindow, center_crop_window
from trajloc.preprocess.stage1.records import (
    identity_error,
    normalized_boxes,
    normalized_positions,
    prompt_record,
    trajectory_entry,
    write_clip_media,
    write_json,
)
from trajloc.preprocess.stage1.sav.masks import merge_parts_into_parents
from trajloc.preprocess.stage1.sav.prompts import build_scene_graph_prompt
from trajloc.preprocess.stage1.sav.selection import select_objects
from trajloc.preprocess.stage1.swarm import SWARM_SIZE, sample_median_neighborhood
from trajloc.preprocess.stage1.tracking import (
    TRACK_LONG_SIDE,
    CoTracker,
    QueryRange,
    Trajectory,
    anchor_trajectory,
)

logger = logging.getLogger(__name__)

MAX_CLIPS_PER_VIDEO = 3
MAX_SRC_FRAMES = MAX_CLIPS_PER_VIDEO * CLIP_NUM_FRAMES  # = 147
MIN_OBJECTS_PER_CLIP = 2
MAX_OBJECTS_PER_CLIP = 15
SCENE_GRAPH_BATCH = 2000  # scene-graph parquet rows read per batch

SHARD_SEED = 20260603  # seeds the assignment of videos to shards


@dataclass(frozen=True)
class SavSource:
    """Where the SA-V release lives (``<root>/train/sav_train/``, ``<root>/svg2/``)."""

    root: Path

    @property
    def scene_graphs(self) -> Path:
        """The SVG2 scene graphs (one parquet row per video)."""
        return self.root / "svg2" / "scene_graphs_sav.parquet"

    @property
    def masks_dir(self) -> Path:
        """The SVG2 mask shards (COCO RLE, one parquet row per video)."""
        return self.root / "svg2" / "masks"

    def video_path(self, video_id: str) -> Path:
        """The video's mp4 (videos are grouped by id prefix)."""
        return (
            self.root
            / "train"
            / "sav_train"
            / ("sav_" + video_id.split("_")[1][:3])
            / f"{video_id}.mp4"
        )


# ---------------------------------------------------------------- SVG2 helpers


@dataclass(frozen=True)
class SceneGraph:
    """A video's SVG2 scene graph: per object, its label (``None`` if it has
    none) and attributes, and the relations ``[src, predicate, dst, _, kind]``."""

    labels: list[str | None]
    attributes: list[list]
    relationships: list

    @classmethod
    def from_row(cls, objects_json: str, relationships_json: str) -> SceneGraph:
        """From a scene-graph parquet row's ``objects`` and ``relationships`` JSON."""
        labels, attributes = parse_objects(objects_json)
        return cls(labels, attributes, json.loads(relationships_json))


def parse_objects(s: str) -> tuple[list[str | None], list[list[Any]]]:
    """An SVG2 ``objects`` column: per object, its label (the last
    ``object_*`` string field, or ``None``) and its attributes."""
    objs = json.loads(s)
    labels: list[str | None] = []
    attrs: list[list[Any]] = []
    for o in objs:
        if not isinstance(o, dict):
            labels.append(None)
            attrs.append([])
            continue
        label = None
        for k, v in o.items():
            if k.startswith("object_") and isinstance(v, str):
                label = v
        labels.append(label)
        attrs.append(o.get("attributes", []) or [])
    return labels, attrs


# ---------------------------------------------------------- per-video pipeline


@dataclass(frozen=True)
class _SourceMasks:
    """A video's SVG2 masks: COCO RLE strings, frame-major
    (``counts[frame * n_obj + obj]``), at ``H x W`` source pixels.

    They decode to uint8 0/1 masks, and tracks stay in the tracker's float32
    when projected (the benchmark builders use bool masks and float64: see
    :class:`~trajloc.preprocess.stage1.geometry.CropWindow`)."""

    counts: list[str]
    num_objects: int
    H: int
    W: int

    def decode(self, frame: int, obj: int) -> np.ndarray | None:
        """The object's uint8 mask on that frame, or ``None`` if it has none
        (missing, empty or undecodable RLE)."""
        flat = frame * self.num_objects + obj
        if flat >= len(self.counts):
            return None
        rle_s = self.counts[flat]
        if not rle_s:
            return None
        try:
            return cocomask.decode({"size": [self.H, self.W], "counts": rle_s.encode("ascii")})
        except Exception as e:  # treated as no mask, as the released data was built
            logger.debug(f"undecodable mask (frame {frame}, object {obj}): {type(e).__name__}: {e}")
            return None


@dataclass(frozen=True)
class _ClipPlan:
    """A clip's objects after selection and swarm seeding: their SVG2
    indices, their swarms (the tracker queries ``(source frame, points)``, in
    object order) and their clip masks (per frame, uint8 or ``None``)."""

    clip_idx: int
    win_start: int
    objects: list[int]
    swarms: list[tuple[int, np.ndarray]]
    clip_masks: list[list[np.ndarray | None]]


class _AnchoredObject(NamedTuple):
    """A planned object whose swarm anchored to a trajectory."""

    source_id: int  # its SVG2 index
    trajectory: Trajectory
    clip_masks: list[np.ndarray | None]


def _read_frames(
    cap, n_frames: int, source_width: int, source_height: int, resize: bool
) -> list[np.ndarray]:
    """Up to ``n_frames`` BGR frames, resized to the masks' size if ``resize``."""
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if resize:
            fr = cv2.resize(fr, (source_width, source_height), interpolation=cv2.INTER_LINEAR)
        frames.append(fr)
        if len(frames) >= n_frames:
            break
    cap.release()
    return frames


def _plan_clip(
    clip_idx: int, masks: _SourceMasks, labels: list[str | None], crop: CropWindow
) -> _ClipPlan | None:
    """Select the clip's objects and seed a swarm for each. ``None`` if fewer
    than ``MIN_OBJECTS_PER_CLIP`` objects are selected, or none is seeded."""
    win_start = clip_idx * CLIP_NUM_FRAMES

    clip_masks_by_object: list[list[np.ndarray | None]] = [
        [None] * CLIP_NUM_FRAMES for _ in range(masks.num_objects)
    ]
    for t in range(CLIP_NUM_FRAMES):
        for oi in range(masks.num_objects):
            m_src = masks.decode(win_start + t, oi)
            if m_src is not None:
                clip_masks_by_object[oi][t] = crop.warp_mask_uint8(m_src)

    merge = merge_parts_into_parents(clip_masks_by_object, labels, H=CLIP_HEIGHT, W=CLIP_WIDTH)
    clip_masks_by_object = merge.masks_by_object
    sel = select_objects(
        labels,
        clip_masks_by_object,
        peak_areas=merge.peak_areas,
        merged=merge.merged,
        H=CLIP_HEIGHT,
        W=CLIP_WIDTH,
        top_n=MAX_OBJECTS_PER_CLIP,
    )
    logger.debug(
        f"selected {sel.selected_indices} (peak pixels {sel.peak_pixels}); rejected "
        f"{sel.rejected}; parts merged into parents {merge.parent_of}"
    )
    # An object need not be visible on the clip's first frame: it is seeded
    # at its largest frame.
    if len(sel.selected_indices) < MIN_OBJECTS_PER_CLIP:
        return None

    swarms = []
    seeded = []
    for source_id in sel.selected_indices:
        swarm = _seed_swarm(masks, win_start, source_id, crop)
        if swarm is not None:
            swarms.append(swarm)
            seeded.append(source_id)

    if not seeded:
        return None
    return _ClipPlan(clip_idx, win_start, seeded, swarms, [clip_masks_by_object[i] for i in seeded])


def _seed_swarm(
    masks: _SourceMasks, win_start: int, source_id: int, crop: CropWindow
) -> tuple[int, np.ndarray] | None:
    """The object's swarm, ``(source frame, (n, 2) source-pixel points)``, or
    ``None`` if it never shows inside the crop window.

    It is seeded on the clip frame with the most mask pixels inside the crop
    window, sampled from those pixels only, so every seed lands in the clip
    frame. CoTracker3 (offline) tracks forwards and backwards from it.
    """
    best_t_local = 0
    best_area = 0
    best_mask: np.ndarray | None = None
    for t in range(CLIP_NUM_FRAMES):
        source_mask = masks.decode(win_start + t, source_id)
        if source_mask is None or source_mask.sum() == 0:
            continue
        mask_in_crop = crop.mask_inside(source_mask)
        area = int(mask_in_crop.sum())
        if area > best_area:
            best_area, best_t_local, best_mask = area, t, mask_in_crop
    if best_mask is None:
        return None  # never inside the crop window: off-screen for the whole clip
    points = sample_median_neighborhood(best_mask, SWARM_SIZE)
    if points.shape[0] == 0:
        return None
    return win_start + best_t_local, points


def _anchor_clip(
    plan: _ClipPlan, source_tracks: np.ndarray, ranges: list[QueryRange], crop: CropWindow
) -> list[_AnchoredObject]:
    """The planned objects whose swarms anchor to a trajectory. ``ranges``
    are the tracker's point ranges of the plan's swarms, in order.

    There is no fallback where the anchor leaves the mask: that is usually a
    brief occlusion CoTracker recovers from, and a mask-median fallback made
    the track jump between anchor and fallback.
    """
    clip_frames = list(range(plan.win_start, plan.win_start + CLIP_NUM_FRAMES))
    anchored = []
    for source_id, r, clip_masks in zip(plan.objects, ranges, plan.clip_masks):
        if r.start == r.end:
            continue  # an empty swarm: nothing was tracked
        tracks_clip_src = source_tracks[clip_frames, r.start : r.end, :]  # (T, K, 2)
        sx_clip, sy_clip = crop.to_clip_pixels(tracks_clip_src[..., 0], tracks_clip_src[..., 1])
        trajectory = anchor_trajectory(sx_clip, sy_clip, clip_masks)
        if trajectory is not None:
            anchored.append(_AnchoredObject(source_id, trajectory, clip_masks))
    return anchored


@dataclass(frozen=True)
class VideoStats:
    """A built video's counts and timings (``timings/<video>.json``)."""

    num_objects: int
    num_clips: int
    stage_seconds: dict[str, float]  # video_read, select_and_seed, cotracker, render_write


@dataclass
class VideoResult:
    """What building one source video produced.

    Attributes:
        clips: per written clip, its summary (``stem``, geometry, kept
            labels, prompt).
        clip_errors: per dropped clip, ``{video_id, clip_idx, error}``.
        video_error: why no clip could be built at all, if so.
        stats: the video's object and clip counts and seconds per stage, if
            it got that far.
    """

    clips: list[dict] = field(default_factory=list)
    clip_errors: list[dict] = field(default_factory=list)
    video_error: dict | None = None
    stats: VideoStats | None = None

    @classmethod
    def failed(cls, video_id: str, error: str) -> VideoResult:
        return cls(video_error={"video_id": video_id, "error": error})


@dataclass(frozen=True)
class _VideoContext:
    """What every clip of a source video is written from."""

    video_id: str
    scene_graph: SceneGraph
    frames: list[np.ndarray]  # the video's BGR frames, at mask resolution
    crop: CropWindow
    source_width: int
    source_height: int
    num_objects: int
    out_dirs: dict[str, Path]

    def emit_clip(
        self, plan: _ClipPlan, anchored: list[_AnchoredObject]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Prompt, check and write one anchored clip. Returns ``(record,
        None)``, its record in ``done/<video>.ok``, or ``(None, error)`` if it
        has too few objects left or fails the identity check."""
        video_id, clip_idx, win_start = self.video_id, plan.clip_idx, plan.win_start
        labels = self.scene_graph.labels
        kept = [obj.source_id for obj in anchored]
        if len(kept) < MIN_OBJECTS_PER_CLIP:
            error = f"too_few_objects_after_anchor:{len(kept)}"
            return None, _clip_error(video_id, clip_idx, error)

        synthesis = build_scene_graph_prompt(
            labels, self.scene_graph.relationships, self.scene_graph.attributes, kept
        )
        trajectory_objects = {
            str(new_idx): trajectory_entry(
                labels[obj.source_id] or "object",
                obj.source_id,
                normalized_positions(*obj.trajectory),
                normalized_boxes(obj.clip_masks),
            )
            for new_idx, obj in enumerate(anchored)
        }
        err = identity_error(synthesis.objects, trajectory_objects)
        if err is not None:
            return None, _clip_error(video_id, clip_idx, f"identity_check_failed:{err}")

        stem = f"{video_id}_c{clip_idx:02d}"
        crop = self.crop
        write_clip_media(
            [crop.warp(f) for f in self.frames[win_start : win_start + CLIP_NUM_FRAMES]],
            self.out_dirs[VIDEOS_DIR] / f"{stem}.mp4",
            self.out_dirs[FIRST_FRAMES_DIR] / f"{stem}.png",
        )
        trajectories = {"objects": trajectory_objects}
        write_json(self.out_dirs[TRAJECTORIES_DIR] / f"{stem}.json", trajectories)
        write_json(self.out_dirs[PROMPTS_DIR] / f"{stem}.json", prompt_record(synthesis))
        return {
            "stem": stem,
            "video_id": video_id,
            "clip_idx": clip_idx,
            "win_start": win_start,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "crop_x0": crop.x0,
            "crop_y0": crop.y0,
            "crop_w": crop.width,
            "crop_h": crop.height,
            "num_objects": self.num_objects,
            "num_objects_kept": len(kept),
            "selected_labels": [labels[i] or "object" for i in kept],
            "prompt": synthesis.prompt,
        }, None


def _clip_error(video_id: str, clip_idx: int, error: str) -> dict[str, Any]:
    """A dropped clip's record (in ``VideoResult.clip_errors``)."""
    return {"video_id": video_id, "clip_idx": clip_idx, "error": error}


def process_video(
    video_id: str,
    mask_row: dict,
    scene_graph: SceneGraph,
    out_dirs: dict[str, Path],
    cotracker: CoTracker,
    source: SavSource,
) -> VideoResult:
    """Build every clip of one source video."""
    masks = _SourceMasks(
        counts=list(mask_row["counts"]),
        num_objects=int(mask_row["num_objects"]),
        H=int(mask_row["mask_height"]),
        W=int(mask_row["mask_width"]),
    )
    source_height, source_width, num_objects = masks.H, masks.W, masks.num_objects
    num_mask_frames = int(mask_row["num_frames"])

    mp4 = source.video_path(video_id)
    if not mp4.exists():
        return VideoResult.failed(video_id, "missing_mp4")

    cap = cv2.VideoCapture(str(mp4))
    num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_video_frames < CLIP_NUM_FRAMES:
        cap.release()
        return VideoResult.failed(video_id, f"too_short={num_video_frames}")

    num_usable_frames = min(num_video_frames, num_mask_frames, MAX_SRC_FRAMES)
    num_clips = num_usable_frames // CLIP_NUM_FRAMES
    if num_clips == 0:
        cap.release()
        return VideoResult.failed(video_id, f"no_full_window n={num_usable_frames}")

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop = center_crop_window(source_width, source_height)

    timings = {}
    t0 = time.perf_counter()
    all_frames = _read_frames(
        cap,
        num_usable_frames,
        source_width,
        source_height,
        resize=(video_width, video_height) != (source_width, source_height),
    )
    timings["video_read"] = time.perf_counter() - t0
    if len(all_frames) < CLIP_NUM_FRAMES:
        return VideoResult.failed(video_id, f"read_short={len(all_frames)}")

    # Select and seed every clip. Prompts are built only after anchoring,
    # which can drop more objects, so the prompt and the trajectories always
    # describe the same objects.
    t0 = time.perf_counter()
    labels = scene_graph.labels
    plans = [_plan_clip(clip_idx, masks, labels, crop) for clip_idx in range(num_clips)]
    queries = [swarm for plan in plans if plan is not None for swarm in plan.swarms]
    timings["select_and_seed"] = time.perf_counter() - t0
    if not any(plan is not None for plan in plans):
        return VideoResult.failed(video_id, "no_kept_clips")

    # Track every clip's swarms in one CoTracker3 pass.
    t0 = time.perf_counter()
    tracked = cotracker.run(all_frames, queries)
    timings["cotracker"] = time.perf_counter() - t0
    if tracked is None:
        return VideoResult.failed(video_id, "cotracker_returned_empty")
    source_tracks, ranges = tracked.tracks, tracked.ranges

    # Anchor, prompt, check and write each clip.
    t0 = time.perf_counter()
    video = _VideoContext(
        video_id=video_id,
        scene_graph=scene_graph,
        frames=all_frames,
        crop=crop,
        source_width=source_width,
        source_height=source_height,
        num_objects=num_objects,
        out_dirs=out_dirs,
    )
    result = VideoResult()
    for plan in plans:
        if plan is None:
            continue
        plan_ranges, ranges = ranges[: len(plan.swarms)], ranges[len(plan.swarms) :]
        record, error = video.emit_clip(plan, _anchor_clip(plan, source_tracks, plan_ranges, crop))
        if error is not None:
            result.clip_errors.append(error)
        else:
            assert record is not None  # emit_clip returns a record or an error
            result.clips.append(record)
    timings["render_write"] = time.perf_counter() - t0
    result.stats = VideoStats(num_objects=num_objects, num_clips=num_clips, stage_seconds=timings)
    return result


# ------------------------------------------------------------------------- run


def shard_video_ids(video_ids: list[str], seed: int, shard: int, num_shards: int) -> list[str]:
    """This shard's videos, sorted. A video's shard is a CRC32 hash of the
    seed and its id: stable across runs, and balanced."""
    seed_b = str(seed).encode("utf-8")
    ids = [
        v
        for v in video_ids
        if (zlib.crc32(seed_b + b"|" + v.encode("utf-8")) % num_shards) == shard
    ]
    ids.sort()
    return ids


def _load_scene_graphs(scene_graph_file: pq.ParquetFile, wanted: set[str]) -> dict[str, SceneGraph]:
    """The scene graphs of the ``wanted`` videos."""
    scene_graphs = {}
    for batch in scene_graph_file.iter_batches(
        batch_size=SCENE_GRAPH_BATCH, columns=["video_id", "objects", "relationships"]
    ):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            if row["video_id"] in wanted:
                scene_graphs[row["video_id"]] = SceneGraph.from_row(
                    row["objects"], row["relationships"]
                )
        if len(scene_graphs) >= len(wanted):
            break
    return scene_graphs


def _iter_mask_rows(source: SavSource, remaining: set[str]):
    """The mask rows of the videos still in ``remaining``, shard file by
    shard file. ``remaining`` is re-read before each file, so videos the
    caller finishes are not looked for again."""
    for shard_path in sorted(source.masks_dir.glob("*.parquet")):
        if not remaining:
            return
        try:
            mask_shard = pq.ParquetFile(shard_path)
        except Exception as e:
            logger.warning(f"unreadable mask shard {shard_path.name}: {e}")
            continue
        ids_in_shard = mask_shard.read(columns=["video_id"]).to_pandas()["video_id"].tolist()
        hits = remaining & set(ids_in_shard)
        if not hits:
            continue
        df = mask_shard.read().to_pandas()
        for _, row in df[df["video_id"].isin(hits)].iterrows():
            yield row


def _write_video_error(out_dirs: dict[str, Path], vid: str, error: dict[str, Any]) -> None:
    """The video's ``errors/<vid>.err`` record."""
    write_json(
        out_dirs[ERRORS_DIR] / f"{vid}.err",
        {"video_id": vid, "step": "extraction", "errors": [error]},
    )


def _record_video(vid: str, result: VideoResult, out_dirs: dict[str, Path], wall: float) -> bool:
    """Write the video's ``errors/<vid>.err`` if no clip could be built, else
    its ``timings/`` and ``done/<vid>.ok`` records. True if done."""
    if result.video_error is not None:
        _write_video_error(out_dirs, vid, result.video_error)
        return False
    if result.stats is not None:
        write_json(
            out_dirs[TIMINGS_DIR] / f"{vid}.json",
            {
                "video_id": vid,
                "num_objects": result.stats.num_objects,
                "num_clips": result.stats.num_clips,
                "num_clips_kept": len(result.clips),
                "stage_seconds": result.stats.stage_seconds,
                "wall_seconds": wall,
            },
        )
    write_json(
        out_dirs[DONE_DIR] / f"{vid}.ok",
        {
            "video_id": vid,
            "num_clips_kept": len(result.clips),
            "num_clip_errors": len(result.clip_errors),
            "clips": result.clips,
        },
    )
    return True


@dataclass
class _RunTally:
    """A run's video counts."""

    done: int = 0  # videos with at least one clip
    errors: int = 0  # videos with none (rejected or failed)
    exceptions: int = 0  # of the errors, those that raised
    no_scene_graph: int = 0  # skipped: SVG2 has no scene graph for them
    clips: int = 0  # clips written


def run(
    *,
    out_dir: Path,
    source: SavSource,
    shard: int = 0,
    num_shards: int = 1,
    seed: int = SHARD_SEED,
    video_ids: list[str] | None = None,
    max_samples: int = 0,
    device: str = "cuda:0",
) -> int:
    """Build clips for this shard's source videos (or ``video_ids``), writing
    ``out_dir/{videos,first_frames,trajectories,prompts}/`` plus
    ``done/<video>.ok``, ``errors/<video>.err`` and ``timings/``. Videos
    with a ``done`` record are skipped, so a run resumes. Returns how many
    videos raised an exception (videos the selection rules reject are not
    failures)."""
    out_root = Path(out_dir)
    out_dirs = {name: out_root / name for name in (*CLIP_DIRS, DONE_DIR, ERRORS_DIR, TIMINGS_DIR)}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"loading CoTracker3 on {device}...")
    t_load = time.time()
    cotracker = CoTracker(device=device, max_resize_long_side=TRACK_LONG_SIDE)
    logger.info(f"loaded in {time.time() - t_load:.1f}s")

    scene_graph_file = pq.ParquetFile(source.scene_graphs)
    all_ids = scene_graph_file.read(columns=["video_id"]).to_pandas()["video_id"].tolist()
    if video_ids:
        sample_ids = list(video_ids)
    else:
        sample_ids = shard_video_ids(all_ids, seed, shard, num_shards)
    if max_samples > 0:
        sample_ids = sample_ids[:max_samples]
    logger.info(f"shard {shard}/{num_shards}: {len(sample_ids)} videos")

    pending = [v for v in sample_ids if not (out_dirs[DONE_DIR] / f"{v}.ok").exists()]
    logger.info(f"resuming: {len(sample_ids) - len(pending)} done, {len(pending)} pending")

    t_scene_graphs = time.time()
    scene_graphs = _load_scene_graphs(scene_graph_file, set(pending))
    logger.info(
        f"scene graphs: {len(scene_graphs)}/{len(pending)} ({time.time() - t_scene_graphs:.1f}s)"
    )

    pending_remaining = set(pending)
    tally = _RunTally()
    t_run = time.time()
    for row in _iter_mask_rows(source, pending_remaining):
        vid = row["video_id"]
        if vid not in scene_graphs:  # SVG2 annotates a subset of the videos
            tally.no_scene_graph += 1
            pending_remaining.discard(vid)
            continue
        t_video = time.time()
        try:
            result = process_video(
                vid,
                row.to_dict(),
                scene_graphs[vid],
                out_dirs,
                cotracker,
                source,
            )
        except Exception as e:  # recorded per video; the run goes on
            msg = f"video_exception:{type(e).__name__}:{e}"
            _write_video_error(out_dirs, vid, {"video_id": vid, "error": msg})
            tally.errors += 1
            tally.exceptions += 1
            pending_remaining.discard(vid)
            continue

        n_clips_kept = len(result.clips)
        if _record_video(vid, result, out_dirs, time.time() - t_video):
            tally.done += 1
            tally.clips += n_clips_kept
        else:
            tally.errors += 1
        pending_remaining.discard(vid)

        processed = tally.done + tally.errors
        rate = processed / (time.time() - t_run)
        eta_hours = max(0, len(pending) - processed) / max(1e-6, rate) / 3600
        logger.info(
            f"  + {vid}  clips={n_clips_kept}  wall={time.time() - t_video:.1f}s  "
            f"({processed}/{len(pending)}, {rate:.2f} v/s, ETA {eta_hours:.2f}h)"
        )

    wall = time.time() - t_run
    logger.info(
        f"shard {shard} done: {tally.done} videos, "
        f"{tally.errors} errors, {tally.clips} clips, "
        f"wall {wall / 3600:.2f}h"
    )
    if tally.no_scene_graph or pending_remaining:
        logger.info(
            f"skipped {tally.no_scene_graph} videos without an SVG2 scene graph and "
            f"{len(pending_remaining)} without a mask row"
        )
    return tally.exceptions
