"""Wan inference: the trained module, a CFG denoising loop, and the VAE decode."""

from __future__ import annotations

import logging
import os
from dataclasses import replace

import numpy as np
import torch

from trajloc.backbones.base import GenerationSettings
from trajloc.backbones.wan import upstream
from trajloc.backbones.wan.checkpoint import WanCheckpointer
from trajloc.backbones.wan.config import WanModelSettings, wan_weights_dir
from trajloc.backbones.wan.constants import LATENT_CHANNELS, WAN_CONFIG_NAME
from trajloc.backbones.wan.model import WanTrajLocModule
from trajloc.core.conditioning import TrajectoryInputs
from trajloc.core.config import TrajLocConfig
from trajloc.core.latent_grid import VAE_SPATIAL_STRIDE, num_latent_frames
from trajloc.datasets.caches import TextCache
from trajloc.datasets.eval_samples import EvalSample

logger = logging.getLogger(__name__)


class WanGenerator:
    """Loads the module + checkpoint and the Wan VAE; generates one clip at a time.

    Args:
        config: the training YAML the checkpoint was trained with.
        settings: checkpoint, device and sampling settings.
    """

    def __init__(self, config: dict, settings: GenerationSettings) -> None:
        self.settings = settings
        self.device = torch.device(settings.device)
        self.model = _load_model(config, settings.checkpoint_dir, self.device)
        self.vae = _load_vae(config, self.device)

    def generate(self, sample: EvalSample, seed: int) -> np.ndarray:
        """``(T, H, W, 3)`` uint8 RGB frames for ``sample``, sampled with ``seed``."""
        latents = self._denoise(TextCache.load(sample.text_cache_path), seed)
        return _decode_latents(latents, self.vae, self.device)

    @torch.no_grad()
    def _denoise(self, cache: TextCache, seed: int) -> torch.Tensor:
        """The CFG denoising loop (text-conditioned vs zero text embedding,
        same image conditioning). Returns latents on CPU."""
        model, device = self.model, self.device
        pipe = model.pipe
        dtype = pipe.torch_dtype

        if cache.image_embeddings is None or cache.i2v_conditioning is None:
            raise ValueError(f"{cache.stem}: not a Wan text cache (no image conditioning)")
        prompt_embedding = cache.prompt_embedding.unsqueeze(0).to(device, dtype=dtype)
        image_embeddings = cache.image_embeddings.unsqueeze(0).to(device, dtype=dtype)
        i2v_conditioning = cache.i2v_conditioning.unsqueeze(0).to(device, dtype=dtype)
        inputs = TrajectoryInputs(
            frame0_latents=model.frame0_latents(i2v_conditioning),
            trajectory_coords=cache.trajectory_coords.unsqueeze(0).to(device, dtype=dtype),
            trajectory_starts=cache.trajectory_starts.unsqueeze(0).to(device, dtype=dtype),
            num_objects=torch.tensor([cache.num_objects]),
            trajectory_token_positions=[cache.trajectory_token_positions],
            noun_token_positions=[cache.noun_token_positions],
        )
        with model.conditioner.conditioned(pipe.dit, prompt_embedding, inputs) as prompt:
            return self._sample(
                pipe, prompt, image_embeddings, i2v_conditioning, model.num_frames, seed
            ).cpu()

    def _sample(
        self, pipe, prompt_embedding, image_embeddings, i2v_conditioning, num_frames: int, seed: int
    ) -> torch.Tensor:
        device, dtype = self.device, pipe.torch_dtype
        height, width = self.model.trajloc_config.video_size
        latents = torch.randn(
            1,
            LATENT_CHANNELS,
            num_latent_frames(num_frames),
            height // VAE_SPATIAL_STRIDE,
            width // VAE_SPATIAL_STRIDE,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )

        pipe.scheduler.set_timesteps(self.settings.num_steps, shift=self.settings.shift)
        uncond_prompt_embedding = torch.zeros_like(prompt_embedding)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

        def velocity(latents, timestep, context):
            return pipe.model_fn(
                **models,
                latents=latents,
                timestep=timestep,
                context=context,
                clip_feature=image_embeddings,
                y=i2v_conditioning,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )

        for timestep in pipe.scheduler.timesteps:
            timestep_t = timestep.unsqueeze(0).to(dtype=dtype, device=device)
            cond = velocity(latents, timestep_t, prompt_embedding)
            uncond = velocity(latents, timestep_t, uncond_prompt_embedding)
            noise_pred = uncond + self.settings.guidance_scale * (cond - uncond)
            latents = pipe.scheduler.step(noise_pred, timestep, latents)
        return latents


def _load_model(config: dict, checkpoint_dir, device: torch.device) -> WanTrajLocModule:
    """The training module, built exactly as for training, with the
    checkpoint loaded, in eval mode."""
    settings = replace(
        WanModelSettings.from_config(config), device=str(device), use_gradient_checkpointing=False
    )
    model = WanTrajLocModule(TrajLocConfig.from_mapping(config["trajloc"]), settings)
    model.to(device)
    model.eval()
    WanCheckpointer.load(model, checkpoint_dir)
    return model


def _load_vae(config: dict, device: torch.device):
    """Wan's VAE, from the config's ``model_path``."""
    upstream.prepare()  # Wan2.1 on sys.path (already done by importing the model)
    from wan.configs import WAN_CONFIGS
    from wan.modules.vae import WanVAE

    wan_config = WAN_CONFIGS[WAN_CONFIG_NAME]
    vae_path = os.path.join(wan_weights_dir(config["model_path"]), wan_config.vae_checkpoint)
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")
    return WanVAE(vae_pth=vae_path, device=str(device))


@torch.no_grad()
def _decode_latents(latents: torch.Tensor, vae, device: torch.device) -> np.ndarray:
    """Wan VAE latents -> ``(T, H, W, 3)`` uint8 RGB."""
    latents_gpu = latents.to(device, dtype=torch.float32)
    video = vae.decode(latents_gpu)  # (1, 3, T, H, W) in [-1, 1]
    video = video[0].permute(1, 2, 3, 0)
    return ((video + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
