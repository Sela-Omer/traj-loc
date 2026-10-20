"""The Wan 2.1 I2V model with TrajLoc's modules, on DiffSynth-Studio.

:class:`WanTrajLocModule` is a DiffSynth ``DiffusionTrainingModule``:

- ``pipe``: the Wan I2V pipeline with only the DiT loaded (text, CLIP and VAE
  inputs come precomputed from the Stage-2 caches), LoRA on the DiT's
  ``q, k, v, o`` projections.
- ``supervisor``: a :class:`~trajloc.backbones.wan.supervisor.WanSupervisor`
  (frozen σ).
- ``appearance_encoder``: an :class:`~trajloc.core.AppearanceEncoder`.
- ``conditioner``: the two above as a
  :class:`~trajloc.core.conditioning.TrajectoryConditioner` (appearance tokens
  written in position order).

Training and inference share it: inference loads a checkpoint into the same
module, so both run the same conditioning code.
"""

from __future__ import annotations

import glob
import logging
from typing import Any

import torch

from trajloc.backbones.wan.config import DIT_WEIGHTS_GLOB, WanModelSettings, wan_weights_dir
from trajloc.backbones.wan.constants import I2V_MASK_CHANNELS
from trajloc.backbones.wan.loss import box_weighted_flow_matching_loss
from trajloc.backbones.wan.supervisor import WanSupervisor
from trajloc.backbones.wan.upstream import DiffusionTrainingModule, ModelConfig, WanVideoPipeline
from trajloc.core.appearance_encoder import AppearanceEncoder, inject_in_position_order
from trajloc.core.conditioning import TrajectoryConditioner, TrajectoryInputs
from trajloc.core.config import TrajLocConfig

logger = logging.getLogger(__name__)


class WanTrajLocModule(DiffusionTrainingModule):
    """Wan I2V + LoRA + trajectory supervision + appearance tokens.

    Args:
        trajloc_config: the config's ``trajloc`` section.
        settings: the pipeline's settings (:class:`WanModelSettings`).
    """

    def __init__(self, trajloc_config: TrajLocConfig, settings: WanModelSettings) -> None:
        DiffusionTrainingModule.__init__(self)
        self.trajloc_config = trajloc_config

        device = settings.device
        weights_dir = wan_weights_dir(settings.model_path)
        logger.info(f"loading the Wan DiT from {weights_dir}")
        # The DiT only: the text and image conditioning come from the Stage-2 caches.
        dit_files = sorted(glob.glob(str(weights_dir / DIT_WEIGHTS_GLOB)))
        model_configs = [ModelConfig(path=dit_files)]

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=None,
        )

        trainable_models = None  # LoRA only
        self.pipe = self.split_pipeline_units(
            settings.task, self.pipe, trainable_models, settings.lora_base_model
        )

        # Adds LoRA to the DiT (initialization draws from the global RNG).
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            settings.lora_base_model,
            settings.lora_target_modules,
            settings.lora_rank,
            None,  # lora_checkpoint
            None,  # preset_lora_path
            None,  # preset_lora_model
            task=settings.task,
        )

        self.use_gradient_checkpointing = settings.use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = False
        self.max_timestep_boundary = 1.0
        self.min_timestep_boundary = 0.0

        self.supervisor = WanSupervisor.from_config(trajloc_config)

        self.appearance_encoder = AppearanceEncoder.from_config(trajloc_config)

        # Not an nn.Module: the two modules above stay this module's children.
        self.conditioner = TrajectoryConditioner(
            self.supervisor, self.appearance_encoder, inject=inject_in_position_order
        )

        logger.info(
            f"box_loss_weight={trajloc_config.box_loss_weight}, "
            f"num_latent_frames={self.supervisor.num_latent_frames}"
        )

    @property
    def num_frames(self) -> int:
        """Video frames per clip."""
        return self.trajloc_config.num_frames

    @staticmethod
    def frame0_latents(i2v_conditioning: torch.Tensor) -> torch.Tensor:
        """The frame-0 VAE latent inside Wan's ``(B, 4 + 16, T, H, W)`` I2V
        conditioning (4 mask channels, then the latents)."""
        return i2v_conditioning[:, I2V_MASK_CHANNELS:, 0, :, :]

    # ---------------------------------------------------------------- training

    def get_pipeline_inputs(self, data: dict[str, Any]) -> tuple[dict, dict, dict]:
        """DiffSynth's ``(shared, positive, negative)`` inputs for a training
        step. Everything is shared (no CFG in training)."""
        shared = {
            "input_latents": data["latents"],
            "context": data["prompt_embeddings"],
            "clip_feature": data["image_embeddings"],
            "y": data["i2v_conditioning"],
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        return shared, {}, {}

    def forward(self, data: dict[str, Any], return_timestep: bool = False):
        """Training loss for a collated batch; ``(loss, timestep)`` with
        ``return_timestep``.

        The supervision stays installed after the forward pass, on purpose
        (see :mod:`trajloc.core.supervision`); the next call replaces it.
        """
        device = self.pipe.device
        dtype = self.pipe.torch_dtype

        latents = data["latents"].to(device, dtype=dtype)
        prompt_embeds = data["prompt_embeddings"].to(device, dtype=dtype)
        trajectory_coords = data["trajectory_coords"].to(device, dtype=dtype)
        trajectory_starts = data["trajectory_starts"].to(device, dtype=dtype)
        image_embeddings = data["image_embeddings"].to(device, dtype=dtype)
        i2v_conditioning = data["i2v_conditioning"].to(device, dtype=dtype)

        prompt_embeds = self.conditioner.condition(
            self.pipe.dit,
            prompt_embeds,
            TrajectoryInputs(
                frame0_latents=self.frame0_latents(i2v_conditioning),
                trajectory_coords=trajectory_coords,
                trajectory_starts=trajectory_starts,
                num_objects=data["num_objects"],
                trajectory_token_positions=data["trajectory_token_positions"],
                noun_token_positions=data["noun_token_positions"],
            ),
        )

        data_processed = {
            "latents": latents,
            "prompt_embeddings": prompt_embeds,
            "image_embeddings": image_embeddings,
            "i2v_conditioning": i2v_conditioning,
        }
        shared, posi, _nega = self.get_pipeline_inputs(data_processed)
        shared, posi, _nega = self.transfer_data_to_device((shared, posi, {}), device, dtype)

        return box_weighted_flow_matching_loss(
            self.pipe,
            box_masks=data.get("box_masks"),
            box_loss_weight=self.trajloc_config.box_loss_weight,
            return_timestep=return_timestep,
            **shared,
            **posi,
        )
