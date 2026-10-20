"""CogVideoX-5B training: the upstream LoRA trainer, extended with TrajLoc.

Batches come from the Stage-2 caches (the prompt embedding is precomputed,
the text encoder is never run). Per step: TrajLoc's conditioning (appearance
tokens, then the joint-attention supervision), CogVideoX's noising and I2V
input, the transformer, and the loss: the v-prediction MSE (weighted by
``1 / (1 - alpha_bar_t)``) blended with its box-masked version
(``box_loss_weight``).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from trajloc.backbones.cogvideox.constants import I2V_IN_CHANNELS
from trajloc.backbones.cogvideox.loss import box_weighted_velocity_loss
from trajloc.backbones.cogvideox.modules import build_conditioner
from trajloc.backbones.cogvideox.upstream import (
    CogVideoXI2VLoraTrainer,
    get_optimizer,
    unwrap_model,
)
from trajloc.backbones.trajloc_modules import (
    load_trajloc_modules,
    save_checkpoint_config,
    save_trajloc_modules,
)
from trajloc.core.conditioning import TrajectoryInputs
from trajloc.core.config import TrajLocConfig
from trajloc.core.latent_grid import num_latent_frames
from trajloc.datasets import build_training_dataset, cached_collate

logger = logging.getLogger(__name__)


class CogVideoXTrajLocTrainer(CogVideoXI2VLoraTrainer):
    """CogVideoX-5B I2V LoRA trainer with TrajLoc's modules.

    Extends the upstream trainer: builds TrajLoc's conditioner (the
    supervisor and appearance encoder) with the model, trains its modules
    with the LoRA, saves them and the training config in every checkpoint
    (:mod:`trajloc.backbones.trajloc_modules`), reads batches from the
    Stage-2 caches, and computes the box-masked loss.

    Args:
        args: the upstream trainer's ``Args``
            (:func:`~trajloc.backbones.cogvideox.training.build_args`).
        config: the training config.
    """

    # Components the upstream trainer unloads: the text encoder (prompt
    # embeddings come from the cache). A list: upstream does
    # ``["transformer"] + self.UNLOAD_LIST``.
    UNLOAD_LIST: ClassVar[list[str]] = ["text_encoder"]

    def __init__(self, args: Any, config: dict[str, Any]) -> None:
        # Before super().__init__, which loads the components.
        self.config = config
        self.trajloc_config = TrajLocConfig.from_mapping(config["trajloc"])
        super().__init__(args)

    def load_components(self):
        """Upstream's components, then TrajLoc's conditioner (see
        :func:`~trajloc.backbones.cogvideox.modules.build_conditioner`)."""
        components = super().load_components()
        self.conditioner = build_conditioner(self.trajloc_config)
        logger.info(
            f"box_loss_weight={self.trajloc_config.box_loss_weight}, "
            f"num_latent_frames={num_latent_frames(self.trajloc_config.num_frames)}"
        )

        return components

    # --------------------------------------------------------- prepare_dataset

    def prepare_dataset(self) -> None:
        """The cached-clip dataset, instead of upstream's video dataset."""
        self.dataset = build_training_dataset(self.trajloc_config, "cogvideox", split="train")
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=cached_collate,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=True,
        )
        logger.info(
            f"prepare_dataset: {len(self.dataset)} samples, "
            f"batch_size={self.args.batch_size}, "
            f"num_workers={self.args.num_workers}"
        )

    # -------------------------------------------- prepare_trainable_parameters

    def prepare_trainable_parameters(self) -> None:
        """Move TrajLoc's modules to the device and hook their save/load
        into accelerate's checkpointing."""
        super().prepare_trainable_parameters()
        self.conditioner.to(self.accelerator.device)
        self.conditioner.appearance_encoder.requires_grad_(True)
        self._register_checkpoint_hooks()

    def prepare_trackers(self) -> None:
        """Upstream's, with the run's settings logged as scalars: TensorBoard's
        hparams take only int, float, str and bool, so the rest (tuples,
        lists, paths, None) are logged as their text."""
        logger.info("Initializing trackers")
        settings = {
            key: value if isinstance(value, (int, float, str, bool)) else str(value)
            for key, value in self.args.model_dump().items()
        }
        self.accelerator.init_trackers(self.args.tracker_name, config=settings)

    def _register_checkpoint_hooks(self) -> None:
        """Save and load TrajLoc's modules (and save the config) alongside
        accelerate's checkpoint."""
        supervisor = self.conditioner.supervisor
        appearance_encoder = self.conditioner.appearance_encoder

        def save_trajloc_modules_hook(_models, _weights, output_dir: str) -> None:
            if self.accelerator.is_main_process:
                save_trajloc_modules(output_dir, supervisor, appearance_encoder)
                save_checkpoint_config(output_dir, self.config)

        def load_trajloc_modules_hook(_models, input_dir: str) -> None:
            load_trajloc_modules(input_dir, supervisor, appearance_encoder)

        self.accelerator.register_save_state_pre_hook(save_trajloc_modules_hook)
        self.accelerator.register_load_state_pre_hook(load_trajloc_modules_hook)

    # ------------------------------------------------------- prepare_optimizer

    def prepare_optimizer(self) -> None:
        """One optimizer param group: the LoRA parameters (fp32), then the
        appearance encoder's (fp32), then σ. The order is the optimizer
        state's layout."""
        from diffusers.optimization import get_scheduler
        from diffusers.training_utils import cast_training_params

        cast_training_params([self.components.transformer], dtype=torch.float32)

        trainable_params: list[torch.Tensor] = []
        trainable_params.extend(
            p for p in self.components.transformer.parameters() if p.requires_grad
        )
        for p in self.conditioner.appearance_encoder.parameters():
            p.data = p.data.float()
            trainable_params.append(p)
        log_sigma = self.conditioner.supervisor.log_sigma
        log_sigma.data = log_sigma.data.float()
        trainable_params.append(log_sigma)

        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_params)
        logger.info(f"Total trainable params: {self.state.num_trainable_parameters:,}")

        params_to_optimize = [{"params": trainable_params, "lr": self.args.learning_rate}]
        self.optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
        )

        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes
        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

    # ------------------------------------------------------------ compute_loss

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """One training step's loss.

        1. TrajLoc's conditioning: the appearance tokens written into the
           prompt embedding, the supervision installed on the transformer
           (replacing the previous step's; it stays installed through the
           backward pass, see :mod:`trajloc.core.supervision`).
        2. Noising: a timestep per sample, then Gaussian noise.
        3. The I2V input: the noisy latents with the (lightly noised) frame-0
           latent, padded to every frame.
        4. The transformer, and the loss (:mod:`~trajloc.backbones.cogvideox.loss`).

        The random draws happen in that order (the appearance encoder's
        reparameterization, the timesteps, the noise, the frame-0 noise).
        """
        device = self.accelerator.device
        dtype = self.state.weight_dtype

        latent = batch["latents"].to(device, dtype=dtype)
        inputs = TrajectoryInputs(
            trajectory_coords=batch["trajectory_coords"].to(device, dtype=dtype),
            trajectory_starts=batch["trajectory_starts"].to(device, dtype=dtype),
            num_objects=batch["num_objects"],
            frame0_latents=batch["frame0_latents"].to(device, dtype=dtype),
            trajectory_token_positions=batch["trajectory_token_positions"],
            noun_token_positions=batch["noun_token_positions"],
        )
        prompt_embedding = self.conditioner.condition(
            unwrap_model(self.accelerator, self.components.transformer),
            batch["prompt_embeddings"].to(device, dtype=dtype),
            inputs,
        )

        latent = self._pad_to_temporal_patch(latent)
        batch_size, _, num_frames, height, width = latent.shape
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=dtype)

        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()
        latent = latent.permute(0, 2, 1, 3, 4)  # (B, F, C, H, W)
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        latent_input = latent_noisy
        if self.state.transformer_config.in_channels == I2V_IN_CHANNELS:
            latent_input = self._i2v_input(latent_noisy, inputs.frame0_latents)

        model_output = self.components.transformer(
            hidden_states=latent_input,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=self._rotary_embeddings(num_frames, height, width),
            return_dict=False,
        )[0]
        return box_weighted_velocity_loss(
            self.components.scheduler,
            model_output,
            latent,
            latent_noisy,
            timesteps,
            batch["box_masks"],
            box_loss_weight=self.trajloc_config.box_loss_weight,
            mask_dtype=self.state.weight_dtype,
        )

    # --------------------------------------------------------------- internals

    def _pad_to_temporal_patch(self, latent: torch.Tensor) -> torch.Tensor:
        """``(B, C, F, H, W)`` latents with frame 0 repeated in front until
        ``F`` is a multiple of the transformer's temporal patch size (the
        released model has none: unchanged)."""
        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            if ncopy > 0:
                first_frame = latent[:, :, :1, :, :]
                latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
        return latent

    def _i2v_input(self, latent_noisy: torch.Tensor, frame0_latents: torch.Tensor) -> torch.Tensor:
        """The I2V transformer's ``(B, F, 2C, H, W)`` input: the noisy latents,
        and the frame-0 latent with noise of std ``exp(N(-3, 0.5^2))`` (one draw
        for the batch) followed by zero frames."""
        device = self.accelerator.device
        dtype = self.state.weight_dtype
        batch_size, num_frames, num_channels, height, width = latent_noisy.shape
        frame0_for_cond = frame0_latents.unsqueeze(2)  # (B, C, 1, H, W)
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=dtype)
        noisy_frame0 = (
            frame0_for_cond
            + torch.randn_like(frame0_for_cond) * image_noise_sigma[:, None, None, None, None]
        )
        image_latents = noisy_frame0.permute(0, 2, 1, 3, 4)  # (B, 1, C, H, W)
        padding_shape = (batch_size, num_frames - 1, num_channels, height, width)
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)
        return torch.cat([latent_noisy, image_latents], dim=2)

    def _rotary_embeddings(self, num_frames: int, height: int, width: int):
        """The transformer's 3D RoPE for a latent clip, or ``None`` if it has none."""
        config = self.state.transformer_config
        if not config.use_rotary_positional_embeddings:
            return None
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        return self.prepare_rotary_positional_embeddings(
            height=height * vae_scale_factor_spatial,
            width=width * vae_scale_factor_spatial,
            num_frames=num_frames,
            transformer_config=config,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            device=self.accelerator.device,
        )
