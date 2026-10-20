"""Benchmark Stage 1B: name every object, write the prompt and trajectories.

Per clip, from Stage 1A's ``objects/<stem>.json`` and crops: name each object
from its highlight crop with Qwen2.5-VL (:mod:`.naming`), build the prompt
over the objects in ``object_index`` order (:mod:`.prompts`), check that the
prompt's objects and the trajectories agree index for index, and write
``trajectories/<stem>.json``, ``prompts/<stem>.json`` and ``done/<stem>.ok``.

Errors are recorded with ``"step": "naming"``.

``prompts/<stem>.json["source_caption"]`` records the caption the benchmark
ships (MoveBench's ``<subset>_track.txt``; MagicBench has none: ``""``). It
is not used for naming; the final dataset appends a scene caption to the
prompt (:mod:`trajloc.preprocess.stage1.caption_join`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image

from trajloc.datasets.layout import (
    DONE_DIR,
    ERRORS_DIR,
    OBJECT_CROPS_DIR,
    PROMPTS_DIR,
    TRAJECTORIES_DIR,
)
from trajloc.preprocess.stage1.bench.benchmarks import ClipSkipped
from trajloc.preprocess.stage1.bench.naming import name_object
from trajloc.preprocess.stage1.bench.prompts import build_benchmark_prompt
from trajloc.preprocess.stage1.qwen import QWEN_MODEL_ID, QwenVL
from trajloc.preprocess.stage1.records import (
    identity_error,
    prompt_record,
    trajectory_entry,
    write_json,
)

logger = logging.getLogger(__name__)


def process_clip(
    qwen: QwenVL,
    stem: str,
    tracked: dict,
    caption: str,
    out_dirs: dict[str, Path],
    crops_root: Path,
) -> dict:
    """Stage 1B for one clip: writes its records and returns a summary.
    Raises :class:`~.benchmarks.ClipSkipped` if its prompt and trajectories
    don't describe the same objects."""
    objects = tracked["objects"]

    descriptions = []
    for record in objects:
        crop_rel = record.get("crop")
        crop_image = None
        if crop_rel:
            p = crops_root / crop_rel
            if p.exists():
                crop_image = Image.open(p).convert("RGB")
        # No scene caption: it biases ambiguous region crops toward the
        # caption's nouns (a mountain named 'hot air balloon').
        object_name = name_object(qwen, crop_image)
        descriptions.append(
            {
                "source_object_id": record["source_object_id"],
                "noun": object_name.noun,
                "phrase": object_name.phrase,
            }
        )

    synthesis = build_benchmark_prompt(descriptions)
    trajectory_objects = {
        str(new_idx): trajectory_entry(
            description["noun"], record["source_object_id"], record["positions"], record["boxes"]
        )
        for new_idx, (record, description) in enumerate(zip(objects, descriptions))
    }
    err = identity_error(synthesis.objects, trajectory_objects)
    if err is not None:
        raise ClipSkipped(err)

    write_json(out_dirs[TRAJECTORIES_DIR] / f"{stem}.json", {"objects": trajectory_objects})
    write_json(
        out_dirs[PROMPTS_DIR] / f"{stem}.json", prompt_record(synthesis, source_caption=caption)
    )

    return {
        "stem": stem,
        "num_kept": len(objects),
        "prompt": synthesis.prompt,
        "words_located": synthesis.ok,
    }


def run(
    stems: list[str],
    out_dir: Path,
    *,
    captions: dict[str, str],
    device: str = "cuda:0",
    model_path: str = QWEN_MODEL_ID,
) -> tuple[int, int]:
    """Stage 1B for ``stems`` of the dataset root ``out_dir`` (clips with a
    ``done`` record, or no ``objects/`` record, are skipped). ``captions``
    maps stem -> the benchmark's caption (missing: ``""``). Returns ``(done,
    failed)``: the clips named (now or before) and those that raised (a
    skipped clip is neither)."""
    done = failed = 0
    out_root = Path(out_dir)
    dirs = (TRAJECTORIES_DIR, PROMPTS_DIR, DONE_DIR, ERRORS_DIR)
    out_dirs = {name: out_root / name for name in dirs}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    crops_root = out_root / OBJECT_CROPS_DIR
    objects_dir = out_root / "objects"

    qwen = QwenVL(model_path, device)

    for stem in stems:
        if (out_dirs[DONE_DIR] / f"{stem}.ok").exists():
            logger.info(f"[{stem}] already done, skipping")
            done += 1
            continue
        objects_path = objects_dir / f"{stem}.json"
        if not objects_path.exists():
            logger.info(f"[{stem}] no Stage-1A objects record, skipping")
            continue
        try:
            with open(objects_path) as f:
                tracked = json.load(f)
            result = process_clip(qwen, stem, tracked, captions.get(stem, ""), out_dirs, crops_root)
        except ClipSkipped as e:
            write_json(
                out_dirs[ERRORS_DIR] / f"{stem}.err",
                {"stem": stem, "error": str(e), "step": "naming"},
            )
            logger.error(f"[{stem}] ERROR: {e}")
            continue
        except Exception as e:  # recorded per clip; the run goes on
            msg = f"{type(e).__name__}:{e}"
            write_json(
                out_dirs[ERRORS_DIR] / f"{stem}.err", {"stem": stem, "error": msg, "step": "naming"}
            )
            logger.error(f"[{stem}] EXCEPTION: {msg}")
            failed += 1
            continue
        write_json(out_dirs[DONE_DIR] / f"{stem}.ok", result)
        (out_dirs[ERRORS_DIR] / f"{stem}.err").unlink(missing_ok=True)  # from an earlier attempt
        logger.info(
            f"[{stem}] OK kept={result['num_kept']} words_located={result['words_located']}"
        )
        logger.info(f"[{stem}] prompt: {result['prompt']}")
        done += 1
    return done, failed
