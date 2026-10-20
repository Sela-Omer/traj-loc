"""Scene captions with Qwen2.5-VL-7B-Instruct (the dataset roots' captions).

Each clip's ``videos/<stem>.mp4`` is sampled at 16 frames (at most
360 * 420 pixels each) and captioned with :data:`CAPTION_PROMPT` by greedy decoding
(max 256 new tokens). The record ``captions/<stem>.json`` is
``{"stem", "caption", "model", "num_frames"}``, written atomically. Clips
with a record are skipped, so a rerun resumes, and retries the clips that
failed (they get no record). The work is split across GPUs statically: clip
``i`` goes to shard ``i % num_shards``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from trajloc.preprocess.io import atomic_write_json
from trajloc.preprocess.sharding import shard_items
from trajloc.preprocess.stage1.qwen import QWEN_MODEL_ID, QwenVL

logger = logging.getLogger(__name__)

# Every captioned dataset uses this prompt and 16 frames, so captions are
# comparable across datasets.
CAPTION_PROMPT = (
    "Describe this video clip in 2-4 sentences. Include all of the following:\n"
    "1. Camera motion — how the camera moves (e.g. stationary, panning left/right, "
    "tracking a subject, slight handheld shake, tilting, zooming in/out, fixed overhead). "
    "Start with this.\n"
    "2. Scene setting — the environment, background, and overall context.\n"
    "3. Each visible subject/object — what it is and what it is doing "
    "(e.g. 'a black swan glides across the water', 'two dogs chase each other across the grass', "
    "'a person on a motorbike performs a jump').\n\n"
    "Be concise and factual. Describe what you see, not what you infer. "
    "Do not speculate about what happens outside the frame.\n\n"
    "Example:\n"
    '"The camera pans slowly to the right, following the action. A sunlit park with tall trees '
    "and a gravel path forms the backdrop. Two dogs run side by side across the grass while "
    'a person in a red jacket walks behind them."'
)


NUM_SAMPLE_FRAMES = 16
MAX_PIXELS_PER_FRAME = 360 * 420
MAX_CAPTION_TOKENS = 256

LOG_FIRST_CLIPS = 3  # progress: the first clips, then every LOG_EVERY_CLIPS
LOG_EVERY_CLIPS = 10


def caption_video(qwen: QwenVL, video_path: Path) -> str:
    """The clip's caption: :data:`CAPTION_PROMPT` over ``NUM_SAMPLE_FRAMES``
    frames, whitespace-normalized, surrounding quotes removed."""
    decoded = qwen.ask_video(
        video_path,
        CAPTION_PROMPT,
        nframes=NUM_SAMPLE_FRAMES,
        max_pixels=MAX_PIXELS_PER_FRAME,
        max_new_tokens=MAX_CAPTION_TOKENS,
    )
    caption = " ".join(decoded.strip().split())
    if caption.startswith('"') and caption.endswith('"'):
        caption = caption[1:-1]
    return caption


def caption_clips(
    videos_dir: Path,
    captions_dir: Path,
    *,
    shard: int = 0,
    num_shards: int = 1,
    max_samples: int = 0,
    device: str = "cuda:0",
    model_path: str = QWEN_MODEL_ID,
) -> int:
    """Caption this shard's clips of ``videos_dir`` that have no record yet
    (at most ``max_samples`` new ones, 0: all); returns how many failed."""
    captions_dir.mkdir(parents=True, exist_ok=True)
    shard_stems = shard_items(sorted(p.stem for p in videos_dir.glob("*.mp4")), shard, num_shards)
    todo = [stem for stem in shard_stems if not (captions_dir / f"{stem}.json").exists()]
    if max_samples > 0:
        todo = todo[:max_samples]
    logger.info(
        f"shard {shard}/{num_shards}: {len(shard_stems)} clips, "
        f"{len(shard_stems) - len(todo)} already captioned, {len(todo)} to do"
    )
    if not todo:
        return 0

    qwen = QwenVL(model_path, device)
    failed = 0
    t_start = time.time()
    for i, stem in enumerate(todo, start=1):
        t0 = time.time()
        try:
            caption = caption_video(qwen, videos_dir / f"{stem}.mp4")
        except Exception as e:  # no record: a rerun retries the clip
            failed += 1
            logger.error(f"{stem}: captioning failed: {type(e).__name__}: {e}")
            continue
        record = {
            "stem": stem,
            "caption": caption,
            "model": qwen.model_path,
            "num_frames": NUM_SAMPLE_FRAMES,
        }
        atomic_write_json(record, captions_dir / f"{stem}.json")
        if i <= LOG_FIRST_CLIPS or i % LOG_EVERY_CLIPS == 0 or i == len(todo):
            rate = i / (time.time() - t_start)
            logger.info(
                f"{i}/{len(todo)} captioned, {failed} failed, {time.time() - t0:.2f}s/clip, "
                f"ETA {(len(todo) - i) / rate / 60:.1f} min"
            )
            logger.info(f"  {stem}: {caption[:140]}{'...' if len(caption) > 140 else ''}")
    minutes = (time.time() - t_start) / 60
    logger.info(f"done: {len(todo) - failed} captioned, {failed} failed, {minutes:.1f} min")
    return failed
