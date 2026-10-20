"""Generate one video from an image, a prompt and trajectories.

The inputs go through exactly the evaluation path, so a generated video is
what the evaluation sets' clips go through:

1. the image and the scene (:mod:`trajloc.generation.scene`) become a
   one-clip Stage-1 dataset root: the image as a still 49-frame clip,
   losslessly encoded, and the ``prompts/`` and ``trajectories/`` records;
2. its Stage-2 caches are built by the Stage-2 steps
   (:class:`~trajloc.preprocess.stage2.latents.LatentCacheStep`,
   :class:`~trajloc.preprocess.stage2.text_cache.TextCacheStep`);
3. the backbone's generator (the one inference uses) generates the clip.

Each stage's models are released before the next is loaded. Output:
``runs/generate/<backbone>/<run_name>/`` with ``video.mp4``,
``generate_config.json`` and the one-clip dataset root under ``inputs/``.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from trajloc.backbones import BackboneSpec, GenerationSettings, get_backbone
from trajloc.datasets.eval_samples import EvalSample
from trajloc.datasets.layout import CLIP_FPS, CLIP_NUM_FRAMES, BackboneName, DatasetRoot
from trajloc.generation.scene import Scene
from trajloc.inference.video import write_video
from trajloc.preprocess.stage1.geometry import center_crop_window
from trajloc.preprocess.stage1.records import write_json
from trajloc.preprocess.stage2.latents import LatentCacheStep
from trajloc.preprocess.stage2.text_cache import TextCacheStep

logger = logging.getLogger(__name__)

STEM = "scene"  # the one clip's stem in the run's dataset root


@dataclass(frozen=True)
class GenerateConfig:
    """One generation.

    Attributes:
        image_path: the first frame (any size: center-cropped to 3:2).
        scene_path: the scene file (:mod:`trajloc.generation.scene`).
        training_config_path: the checkpoint's training config.
        checkpoint_dir: a TrajLoc checkpoint directory.
        trajectory_encoder_path: the frozen trajectory encoder's weights.
        run_dir: the output directory (must not exist).
        device: torch device string.
        guidance_scale: classifier-free guidance scale.
        num_steps: sampling steps.
        shift: flow-matching scheduler shift (``None`` for backbones without one).
        seed: seeds the Stage-2 encodings and the sampling.
    """

    image_path: Path
    scene_path: Path
    training_config_path: Path
    checkpoint_dir: Path
    trajectory_encoder_path: Path
    run_dir: Path
    device: str
    guidance_scale: float
    num_steps: int
    shift: float | None
    seed: int


def generate(backbone: BackboneName, scene: Scene, run: GenerateConfig) -> Path:
    """Generate the scene's video; returns its path (``<run_dir>/video.mp4``).

    The inputs are checked (the image is read and the scene mapped into its
    crop: :class:`~trajloc.generation.scene.SceneError` or
    ``FileNotFoundError``) before the run directory is created; a run that
    fails or is interrupted removes it, so its ``--run_name`` can be reused.
    """
    spec = get_backbone(backbone)
    clip = SceneClip.of(scene, run.image_path)
    run.run_dir.mkdir(parents=True)
    try:
        return _generate_into(spec, backbone, scene, clip, run)
    except BaseException:
        shutil.rmtree(run.run_dir, ignore_errors=True)
        raise


def _generate_into(
    spec: BackboneSpec, backbone: BackboneName, scene: Scene, clip: SceneClip, run: GenerateConfig
) -> Path:
    """:func:`generate`'s work, in the (new) run directory."""
    _save_config(run)
    config = yaml.safe_load(run.training_config_path.read_text())

    root = DatasetRoot(run.run_dir / "inputs")
    write_scene_clip(root, clip)
    logger.info(f"scene: {len(scene.objects)} objects; building its Stage-2 caches")

    torch.manual_seed(run.seed)  # CogVideoX's VAE samples its latents
    stage2_steps = (
        lambda: LatentCacheStep(root.path, backbone, device=run.device),
        lambda: TextCacheStep(
            root.path,
            backbone,
            dataset_name="generate",
            device=run.device,
            trajectory_encoder_path=run.trajectory_encoder_path,
        ),
    )
    for make_step in stage2_steps:  # one at a time, so each step's models are freed
        step = make_step()
        failed, name = step.run(), step.name
        del step
        _release_gpu_memory()
        if failed:  # the step logged the reason (a too-long prompt, say)
            raise ValueError(f"{name} failed on the scene: see the error logged above")

    generator = spec.load_generator(
        config,
        GenerationSettings(
            checkpoint_dir=run.checkpoint_dir,
            device=run.device,
            guidance_scale=run.guidance_scale,
            num_steps=run.num_steps,
            shift=run.shift,
        ),
    )
    files = root.clip_files(STEM)
    sample = EvalSample(
        eval_set="generate",
        stem=STEM,
        index=0,
        text_cache_path=str(root.text_cache_dir(backbone) / f"{STEM}.pt"),
        latent_path=str(root.latent_cache_dir(backbone) / f"{STEM}.pt"),
        video_path=str(files.video),
    )
    video_path = run.run_dir / "video.mp4"
    write_video(video_path, generator.generate(sample, seed=run.seed))
    logger.info(f"wrote {video_path}")
    return video_path


@dataclass(frozen=True)
class SceneClip:
    """A scene's clip, ready to write: its first frame (BGR, cropped and
    resized to the clip size) and its Stage-1 records."""

    frame_bgr: np.ndarray[Any, np.dtype[np.uint8]]
    prompts: dict[str, Any]
    trajectories: dict[str, Any]

    @classmethod
    def of(cls, scene: Scene, image_path: Path) -> SceneClip:
        """Read the image and map the scene into its center crop."""
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"cannot read an image at {image_path}")
        height, width = image_bgr.shape[:2]
        prompts, trajectories = scene.records(width, height)
        return cls(center_crop_window(width, height).warp(image_bgr), prompts, trajectories)


def write_scene_clip(root: DatasetRoot, clip: SceneClip) -> None:
    """The clip as a one-clip Stage-1 dataset root: its frame as a still clip
    (``videos/``, losslessly encoded so every reader sees its exact pixels;
    ``first_frames/``), and its ``prompts/`` and ``trajectories/`` records."""
    frame_bgr = clip.frame_bgr

    for directory in (
        root.videos_dir,
        root.first_frames_dir,
        root.trajectories_dir,
        root.prompts_dir,
    ):
        directory.mkdir(parents=True)
    files = root.clip_files(STEM)
    _write_still_clip(files.video, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cv2.imwrite(str(files.first_frame), frame_bgr)
    write_json(files.prompts, clip.prompts)
    write_json(files.trajectories, clip.trajectories)


def _write_still_clip(path: Path, frame_rgb: np.ndarray[Any, np.dtype[np.uint8]]) -> None:
    """``frame_rgb`` repeated for a clip's frames, as a lossless RGB H.264 mp4."""
    import av

    height, width = frame_rgb.shape[:2]
    frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264rgb", rate=CLIP_FPS)
        if not isinstance(stream, av.VideoStream):
            raise TypeError(f"libx264rgb gave a {type(stream).__name__}, not a video stream")
        stream.width, stream.height, stream.pix_fmt = width, height, "rgb24"
        stream.options = {"crf": "0"}  # lossless
        for _ in range(CLIP_NUM_FRAMES):
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def _save_config(run: GenerateConfig) -> None:
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(run).items()}
    path = run.run_dir / "generate_config.json"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _release_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
