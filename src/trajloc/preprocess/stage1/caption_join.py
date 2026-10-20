"""Scene captions appended to object prompts: a dataset's final prompts.

Stage 1 writes, per clip, an *object prompt* that names each tracked object
next to its ``<traj_i>`` token, e.g. ``"A dog <traj_0> chasing a ball
<traj_1>."``. A dataset's final prompts append a free-form scene caption to
it::

    prompt = object_prompt + " " + scene_caption

Only the top-level ``prompt`` string changes. ``objects[]`` is copied
verbatim, and because the caption is only *appended*, every word index in
``objects[]`` (``noun_word_index``, ``trajectory_token_word_index``) still
points at the same word. So the final dataset root shares the base root's
videos, trajectories and VAE latents (symlinked); only its text caches are
computed anew.

The three datasets differ in the details:

==========  =======================  ===============  ================
dataset     scene caption            cleaned          empty caption
==========  =======================  ===============  ================
SA-V        Qwen2.5-VL, captions/    afterwards [#]_  still appended
MoveBench   the benchmark's own      at join time     prompt unchanged
MagicBench  Qwen2.5-VL, captions/    at join time     prompt unchanged
==========  =======================  ===============  ================

.. [#] SA-V is joined with its raw captions; :func:`normalize_captions` then
   rewrites, in place, the few whose characters Wan's tokenizer normalizes
   (see :mod:`trajloc.preprocess.text`).

The joined record keeps both halves of the prompt: ``scene_caption`` and
``object_prompt``.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from trajloc.datasets.layout import (
    BACKBONE_NAMES,
    CAPTIONS_DIR,
    FIRST_FRAMES_DIR,
    PROMPTS_DIR,
    SPLITS_FILE,
    TRAJECTORIES_DIR,
    VIDEOS_DIR,
    DatasetRoot,
    cache_subdir,
)
from trajloc.preprocess.io import atomic_write_json
from trajloc.preprocess.stage1.errors import Stage1Error
from trajloc.preprocess.text import basic_clean

logger = logging.getLogger(__name__)


def joined_prompt_record(base_meta: dict, caption: str, *, keep_prompt_if_no_caption: bool) -> dict:
    """The joined ``prompts/<stem>.json`` record for one clip.

    Args:
        base_meta: the base dataset's prompt record (``prompt``, ``objects``,
            optional ``words_located`` / ``word_location_errors``).
        caption: the scene caption to append, already cleaned if it should be.
        keep_prompt_if_no_caption: if the caption is blank, keep the object
            prompt as is instead of appending ``" " + caption``.
    """
    base_prompt = base_meta["prompt"]
    if keep_prompt_if_no_caption and not caption.strip():
        prompt = base_prompt
    else:
        prompt = base_prompt + " " + caption
    return {
        "prompt": prompt,
        "objects": base_meta["objects"],
        "words_located": base_meta.get("words_located", True),
        "word_location_errors": base_meta.get("word_location_errors", []),
        "scene_caption": caption,
        "object_prompt": base_prompt,
    }


# ------------------------------------------------------------ shared layout

# What a final root shares with its base root, as symlinks (none of it
# depends on the prompt).
SHARED_DIRS = (
    VIDEOS_DIR,
    FIRST_FRAMES_DIR,
    TRAJECTORIES_DIR,
    *(cache_subdir(backbone, "latents") for backbone in BACKBONE_NAMES),
)
SHARED_FILES = (SPLITS_FILE,)
# The text caches depend on the prompts, so each root has its own.
TEXT_CACHE_DIRS = tuple(cache_subdir(backbone, "text") for backbone in BACKBONE_NAMES)


def ensure_symlink(src: Path, dst: Path) -> None:
    """Make ``dst`` a symlink to ``src``'s absolute path. A no-op if it
    already is. Refuses to retarget a symlink or replace a real path."""
    src_abs = src.resolve()
    if dst.is_symlink():
        existing = os.readlink(dst)
        if Path(existing).resolve() == src_abs:
            return
        raise Stage1Error(f"symlink {dst} already points elsewhere: {existing} (want {src_abs})")
    if dst.exists():
        raise Stage1Error(f"refusing to overwrite real path at {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(src_abs), str(dst))


def link_shared_layout(base_root: Path, out_root: Path) -> None:
    """Create the final root's directory tree and link what it shares with
    the base root. Needs no captions, so it can run while captioning is in
    progress."""
    logger.info("=== step: symlinks ===")
    for sub in ("", PROMPTS_DIR, CAPTIONS_DIR, "cache"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    for rel in SHARED_DIRS + SHARED_FILES:
        src, dst = base_root / rel, out_root / rel
        if not src.exists():
            logger.info(f"  skip {rel}: source missing at {src}")
            continue
        ensure_symlink(src, dst)
        logger.info(f"  symlink {dst} -> {src}")
    for sub in TEXT_CACHE_DIRS:
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    logger.info("symlinks step done.")


# --------------------------------------------------------------------- SA-V

# A clip's join status (anything else is an error, reported by its name).
WROTE = "wrote"
SKIPPED_EXISTING = "skip_exists"
_OK_STATUSES = (WROTE, SKIPPED_EXISTING)


def _join_sav_clip(
    stem: str, base_prompts_dir: Path, captions_dir: Path, out_dir: Path
) -> tuple[str, str, str]:
    """Worker: write one joined prompt. Returns ``(stem, status, detail)``."""
    out_path = out_dir / f"{stem}.json"
    if out_path.exists():
        return stem, SKIPPED_EXISTING, ""
    prompt_path = base_prompts_dir / f"{stem}.json"
    caption_path = captions_dir / f"{stem}.json"
    if not prompt_path.exists():
        return stem, "missing_prompt", ""
    if not caption_path.exists():
        return stem, "missing_caption", ""
    try:
        with open(prompt_path) as f:
            base_meta = json.load(f)
        with open(caption_path) as f:
            caption_meta = json.load(f)
    except Exception as e:
        return stem, "read_error", f"{type(e).__name__}: {e}"
    # A record missing "prompt" or "caption" raises, aborting the run.
    record = joined_prompt_record(
        base_meta, caption_meta["caption"], keep_prompt_if_no_caption=False
    )
    try:
        atomic_write_json(record, out_path)
    except Exception as e:
        return stem, "write_error", f"{type(e).__name__}: {e}"
    return stem, WROTE, ""


def join_sav_captions(
    base_root: Path, out_root: Path, *, workers: int, allow_partial: bool, stems: list[str] | None
) -> int:
    """Write ``out_root/prompts/<stem>.json`` for every base clip not yet joined.

    Clips without a caption abort the run (:class:`Stage1Error`) unless
    ``allow_partial``. If ``stems`` is given (even empty), only those clips
    are considered. Returns the exit code (1 if any clip failed).
    """
    logger.info("=== step: prompts ===")
    base_prompts_dir = DatasetRoot(base_root).prompts_dir
    captions_dir = DatasetRoot(out_root).captions_dir
    out_dir = DatasetRoot(out_root).prompts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_stems = sorted(p.stem for p in base_prompts_dir.glob("*.json"))
    base_stem_set = set(base_stems)
    captioned = {p.stem for p in captions_dir.glob("*.json")}
    joined = {p.stem for p in out_dir.glob("*.json")}

    if stems is not None:
        unknown = [s for s in stems if s not in base_stem_set]
        if unknown:
            raise Stage1Error(
                f"{len(unknown)} requested stems not in the base root (first 5): {unknown[:5]}"
            )
        todo = [s for s in stems if s not in joined]
        logger.info(
            f"--stems selection: {len(stems)} requested, {len(stems) - len(todo)} already done"
        )
    else:
        todo = [s for s in base_stems if s not in joined]

    logger.info(f"object prompts:  {len(base_stems)}")
    logger.info(f"scene captions:  {len(captioned)}")
    logger.info(f"joined prompts:  {len(joined)}")
    logger.info(f"todo this run:     {len(todo)}")

    uncaptioned = [s for s in todo if s not in captioned]
    if uncaptioned and not allow_partial:
        raise Stage1Error(
            f"{len(uncaptioned)} stems lack a caption (first 10): {uncaptioned[:10]}"
            " (re-run with --allow_partial to skip them)"
        )
    if uncaptioned:
        logger.warning(f"{len(uncaptioned)} stems lack a caption; skipping them")
        todo = [s for s in todo if s in captioned]
        logger.info(f"todo after filtering: {len(todo)}")
    if not todo:
        logger.info("no prompts to write.")
        return 0

    logger.info(f"writing {len(todo)} prompts (workers={workers})")
    counts: dict[str, int] = {}
    first_errors: list[tuple[str, str, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_join_sav_clip, s, base_prompts_dir, captions_dir, out_dir) for s in todo
        ]
        for n, fut in enumerate(as_completed(futures), start=1):
            stem, status, detail = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status not in _OK_STATUSES and len(first_errors) < 10:
                first_errors.append((stem, status, detail))
            if n % 5000 == 0 or n == len(futures):
                logger.info(f"  progress {n}/{len(futures)}  counts={counts}")
    if first_errors:
        logger.info("first errors:")
        for stem, status, detail in first_errors:
            logger.info(f"  {stem}: {status}  {detail}")
    return 1 if any(k not in _OK_STATUSES for k in counts) else 0


def normalize_captions(root: Path, *, apply: bool, stems: list[str] | None) -> None:
    """Clean a joined root's scene captions that Wan's tokenizer would
    otherwise change, in place.

    For each caption where ``basic_clean(caption) != caption``: rewrite the
    caption file and the joined prompt (from the record's own
    ``object_prompt``) with the cleaned caption, and delete both text caches
    of the clip so Stage 2 recomputes them. Without ``apply`` it only
    reports. If ``stems`` is given (even empty), only those clips are
    checked. Idempotent.
    """
    captions_dir = DatasetRoot(root).captions_dir
    prompts_dir = DatasetRoot(root).prompts_dir
    text_cache_dirs = [root / d for d in TEXT_CACHE_DIRS]
    logger.info(f"root: {root}")

    affected: list[tuple[str, str, str]] = []  # (stem, before, after)
    if stems is not None:
        logger.info(f"inspecting {len(stems)} known stems (--stems)")
        paths = []
        for stem in stems:
            path = captions_dir / f"{stem}.json"
            if path.exists():
                paths.append(path)
            else:
                logger.warning(f"  {stem}: caption missing")
    else:
        logger.info(f"scanning {captions_dir} for affected captions...")
        paths = [p for p in sorted(captions_dir.glob("*.json")) if not p.name.startswith("_")]
    for path in paths:
        with open(path) as f:
            caption = json.load(f).get("caption", "")
        cleaned = basic_clean(caption)
        if cleaned != caption:
            affected.append((path.stem, caption, cleaned))
        elif stems is not None:
            logger.info(f"  {path.stem}: clean already; no change needed")
    logger.info(f"checked {len(paths)} captions; {len(affected)} affected")
    if not affected:
        logger.info("nothing to fix.")
        return

    for stem, before, after in affected:
        _log_first_change(stem, before, after)
    if not apply:
        logger.info(
            "dry run: re-run with --apply to rewrite captions/prompts "
            "and delete the stale text caches."
        )
        return

    logger.info("applying...")
    n_captions = n_prompts = n_caches = 0
    for stem, _before, after in affected:
        caption_path = captions_dir / f"{stem}.json"
        with open(caption_path) as f:
            caption_record = json.load(f)
        if caption_record.get("caption") != after:
            caption_record["caption"] = after
            atomic_write_json(caption_record, caption_path)
            n_captions += 1

        prompt_path = prompts_dir / f"{stem}.json"
        with open(prompt_path) as f:
            prompt_record = json.load(f)
        if "object_prompt" not in prompt_record:  # kept as the released data was built
            logger.warning(f"{prompt_path}: no object_prompt; the prompt is the caption alone")
        joined = prompt_record.get("object_prompt", "") + " " + after
        if prompt_record.get("prompt") != joined or prompt_record.get("scene_caption") != after:
            prompt_record["prompt"] = joined
            prompt_record["scene_caption"] = after
            atomic_write_json(prompt_record, prompt_path)
            n_prompts += 1

        for cache_dir in text_cache_dirs:
            cache = cache_dir / f"{stem}.pt"
            if cache.exists():
                cache.unlink()
                n_caches += 1
    logger.info(f"  rewrote captions: {n_captions}")
    logger.info(f"  rewrote prompts:  {n_prompts}")
    logger.info(f"  deleted text caches: {n_caches}")
    logger.info("done. Re-run the Stage-2 text caches for the deleted ones.")


def _log_first_change(stem: str, before: str, after: str) -> None:
    """Log where a caption fix first changes the caption, with context."""
    for i, (a, b) in enumerate(zip(before, after, strict=False)):
        if a != b:
            lo, hi = max(0, i - 30), min(len(before), i + 30)
            logger.info(f"  {stem}: char {i}: {a!r} (U+{ord(a):04X}) -> {b!r} (U+{ord(b):04X})")
            logger.info(f"    before: ...{before[lo:hi]}...")
            logger.info(f"    after:  ...{after[lo:hi]}...")
            return
    logger.info(f"  {stem}: length changed: {len(before)} -> {len(after)}")


# ---------------------------------------------------------------- benchmarks


def join_benchmark_captions(base_root: Path, out_root: Path, *, captions_dir: Path | None) -> int:
    """Write ``out_root/prompts/<stem>.json`` for every base clip.

    The caption comes from ``captions_dir/<stem>.json["caption"]`` if given
    (MagicBench: every clip must have one), else from the base record's own
    ``source_caption`` (MoveBench ships a caption per clip). It is cleaned
    before joining. Returns the exit code (1 if any clip failed).
    """
    base_prompts = DatasetRoot(base_root).prompts_dir
    out_prompts = DatasetRoot(out_root).prompts_dir
    out_prompts.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in base_prompts.glob("*.json"))
    if not stems:
        raise Stage1Error(f"no base prompts found under {base_prompts}")
    if captions_dir is not None:
        captioned = {p.stem for p in captions_dir.glob("*.json")}
        missing = [s for s in stems if s not in captioned]
        if missing:
            raise Stage1Error(
                f"{len(missing)} clips lack a caption: {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}"
            )

    n_wrote = n_err = n_cleaned = n_empty = 0
    for stem in stems:
        try:
            base_meta = json.loads((base_prompts / f"{stem}.json").read_text())
            caption_meta = (
                json.loads((captions_dir / f"{stem}.json").read_text())
                if captions_dir is not None
                else None
            )
        except Exception as e:
            logger.error(f"  FAIL {stem}: read_error: {type(e).__name__}: {e}")
            n_err += 1
            continue
        if caption_meta is not None:
            raw = caption_meta.get("caption", "") or ""
        else:
            raw = base_meta.get("source_caption", "") or ""
        caption = basic_clean(raw)
        if caption != raw:
            n_cleaned += 1
        if not caption.strip():
            n_empty += 1
        record = joined_prompt_record(base_meta, caption, keep_prompt_if_no_caption=True)
        try:
            atomic_write_json(record, out_prompts / f"{stem}.json")
        except Exception as e:
            logger.error(f"  FAIL {stem}: write_error: {type(e).__name__}: {e}")
            n_err += 1
            continue
        n_wrote += 1
        logger.info(
            f"  joined {stem}: {len(base_meta['prompt'])}+{len(caption)} chars -> "
            f"{len(record['prompt'])} chars"
        )

    logger.info(
        f"wrote={n_wrote} errors={n_err} total={len(stems)} "
        f"ftfy_cleaned={n_cleaned} empty_caption={n_empty}"
    )
    return 1 if n_err else 0
