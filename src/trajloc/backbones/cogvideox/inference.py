"""CogVideoX inference: the diffusers I2V pipeline with the LoRA fused and
TrajLoc's modules attached.

Pipeline setup:
- ``CogVideoXImageToVideoPipeline`` in bf16, the LoRA loaded and *fused*
  (scale 1.0);
- the scheduler rebuilt with ``timestep_spacing="trailing"``;
- VAE slicing and tiling on;
- dynamic CFG at sampling time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from trajloc.backbones.base import GenerationSettings
from trajloc.backbones.cogvideox.constants import LATENT_SCALE, MODEL_ID, MODEL_REVISION
from trajloc.backbones.cogvideox.modules import build_conditioner
from trajloc.backbones.cogvideox.training import CONFIG_KEYS
from trajloc.backbones.trajloc_modules import load_trajloc_modules
from trajloc.core.conditioning import TrajectoryInputs
from trajloc.core.config import TrajLocConfig, check_config_keys
from trajloc.datasets.caches import LatentCache, TextCache
from trajloc.datasets.eval_samples import EvalSample

logger = logging.getLogger(__name__)

LORA_FILE = "pytorch_lora_weights.safetensors"  # diffusers' name: loadable by diffusers directly


class CogVideoXGenerator:
    """Loads the pipeline + checkpoint; generates one clip at a time.

    Args:
        config: the training YAML the checkpoint was trained with.
        settings: checkpoint, device and sampling settings (no shift).
    """

    def __init__(self, config: dict, settings: GenerationSettings) -> None:
        self.settings = settings
        self.device = torch.device(settings.device)
        check_config_keys(config, CONFIG_KEYS, "CogVideoX")
        trajloc_config = TrajLocConfig.from_mapping(config["trajloc"])
        self.pipe = _load_pipeline(config, settings.checkpoint_dir, self.device)
        self.conditioner = build_conditioner(trajloc_config)
        load_trajloc_modules(
            settings.checkpoint_dir,
            self.conditioner.supervisor,
            self.conditioner.appearance_encoder,
        )
        self.conditioner.to(self.device).eval()
        self.num_frames = trajloc_config.num_frames
        self.video_size = trajloc_config.video_size

    @torch.no_grad()
    def generate(self, sample: EvalSample, seed: int) -> np.ndarray:
        """``(T, H, W, 3)`` uint8 RGB frames for ``sample``, sampled with ``seed``."""
        cache = TextCache.load(sample.text_cache_path)
        frame0_latent = LatentCache.load(sample.latent_path).frame0_latent
        pipe, device = self.pipe, self.device
        clip_height, clip_width = self.video_size

        frame0_pil = _conditioning_image(sample.video_path, frame0_latent, pipe, device)
        prompt_embedding = cache.prompt_embedding.unsqueeze(0).to(device, dtype=torch.bfloat16)
        inputs = TrajectoryInputs(
            frame0_latents=frame0_latent.unsqueeze(0).to(device, dtype=torch.float32),
            # bf16, unlike its neighbors: the dtype the released model was evaluated with.
            trajectory_coords=cache.trajectory_coords.unsqueeze(0).to(device, dtype=torch.bfloat16),
            trajectory_starts=cache.trajectory_starts.unsqueeze(0).to(device, dtype=torch.float32),
            num_objects=torch.tensor([cache.num_objects]),
            trajectory_token_positions=[cache.trajectory_token_positions],
            noun_token_positions=[cache.noun_token_positions],
        )
        with self.conditioner.conditioned(pipe.transformer, prompt_embedding, inputs) as prompt:
            video_output = pipe(
                image=frame0_pil,
                prompt_embeds=prompt,
                negative_prompt_embeds=torch.zeros_like(prompt),
                num_frames=self.num_frames,
                height=clip_height,
                width=clip_width,
                num_inference_steps=self.settings.num_steps,
                guidance_scale=self.settings.guidance_scale,
                use_dynamic_cfg=True,
                generator=torch.Generator(device=device).manual_seed(seed),
            ).frames[0]

        return np.array([np.array(f) for f in video_output])


def _load_pipeline(config: dict, checkpoint_dir, device: torch.device):
    """The I2V pipeline with the checkpoint's LoRA fused, on ``device``."""
    from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline

    model_path = config["model_path"]
    ckpt_dir = Path(checkpoint_dir)

    logger.info(f"Loading CogVideoX-5B-I2V pipeline from {model_path}")
    revision = MODEL_REVISION if model_path == MODEL_ID else None  # a local path: as is
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_path, revision=revision, torch_dtype=torch.bfloat16
    )

    lora_path = ckpt_dir / LORA_FILE
    if not lora_path.exists():
        raise FileNotFoundError(f"No LoRA weights at {lora_path}")
    pipe.load_lora_weights(str(ckpt_dir), weight_name=lora_path.name, adapter_name="trajloc")
    pipe.fuse_lora(components=["transformer"], lora_scale=1.0)
    logger.info(f"LoRA loaded + fused from {lora_path}")

    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def _conditioning_image(video_path: str | None, frame0_latent: torch.Tensor, pipe, device):
    """The I2V conditioning image: the GT video's first frame, or, without
    a GT video, the frame-0 latent decoded by the pipeline's VAE."""
    from PIL import Image

    if video_path is not None and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # A documented behavior (trajloc.backbones): the decoded latent differs
    # slightly from the video's frame, so the results do too.
    logger.warning(f"no readable video at {video_path}: conditioning on the decoded frame-0 latent")
    return _decode_frame0_from_latent(frame0_latent, pipe, device)


def _decode_frame0_from_latent(frame0_latent: torch.Tensor, pipe, device):
    """``(16, H, W)`` frame-0 latent -> PIL image. The latents were stored at
    ``LATENT_SCALE`` (not the pipeline config's ``scaling_factor``)."""
    from PIL import Image

    latent = frame0_latent.unsqueeze(0).unsqueeze(2).to(device, dtype=torch.bfloat16) / LATENT_SCALE
    decoded = pipe.vae.decode(latent).sample  # (B, 3, T, H, W)
    frame0_image = decoded[0, :, 0].permute(1, 2, 0).float().cpu().numpy()
    frame0_image = ((frame0_image + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(frame0_image)
