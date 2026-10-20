"""CogVideoX's Stage-2 encoders: the VAE latents and the T5 text caches.

- :class:`CogVideoXVideoEncoder`: CogVideoX-5B's VAE (bf16, tiling + slicing),
  latents scaled by ``LATENT_SCALE``. ``frame0_latent`` is frame 0 encoded as
  a 49-frame clip of repeated frame 0 (so the temporal convolutions see no
  other frame), first latent. Both encodes *sample* the latent distribution
  with the global RNG.
- :class:`CogVideoXPromptEncoder`: the CogVideoX-5B-I2V T5 tokenizer (with the
  ``<traj_i>`` tokens added; padded/truncated to ``TEXT_SEQ_LENGTH``) and T5
  encoder (fp16).

The encoders are the base models' (``VAE_MODEL_ID``, ``MODEL_ID``, at pinned revisions:
the caches are part of the data, not of a training config).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from trajloc.backbones.base import TokenizedPrompt
from trajloc.backbones.cogvideox.constants import (
    LATENT_SCALE,
    MODEL_ID,
    MODEL_REVISION,
    TEXT_SEQ_LENGTH,
    TEXT_TOKEN_STD,
    VAE_MODEL_ID,
    VAE_MODEL_REVISION,
)
from trajloc.core.trajectory_encoder import trajectory_token_strings
from trajloc.datasets.caches import LatentCache
from trajloc.datasets.layout import CLIP_NUM_FRAMES, ClipFiles

logger = logging.getLogger(__name__)


class CogVideoXVideoEncoder:
    """CogVideoX-5B VAE latents (``cache/cogvideox/latents/``)."""

    def load(self, device: str) -> None:
        from diffusers import AutoencoderKLCogVideoX

        logger.info("loading the CogVideoX-5B VAE")
        vae = (
            AutoencoderKLCogVideoX.from_pretrained(
                VAE_MODEL_ID,
                subfolder="vae",
                revision=VAE_MODEL_REVISION,
                torch_dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        vae.enable_tiling()
        vae.enable_slicing()
        self.vae = vae
        self.device = device

    @torch.no_grad()
    def encode(self, video: torch.Tensor, source_video: str) -> LatentCache:
        clip = video.permute(1, 0, 2, 3).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        latents = self.vae.encode(clip).latent_dist.sample() * LATENT_SCALE
        latents = latents.squeeze(0).cpu().float()

        # Frame 0 alone: [frame 0] x 49, so no other frame leaks in.
        frame0_clip = video[0:1].repeat(CLIP_NUM_FRAMES, 1, 1, 1)
        frame0_clip = frame0_clip.permute(1, 0, 2, 3).unsqueeze(0)
        frame0_clip = frame0_clip.to(self.device, dtype=torch.bfloat16)
        frame0_latents = self.vae.encode(frame0_clip).latent_dist.sample() * LATENT_SCALE
        frame0_latent = frame0_latents.squeeze(0)[:, 0, :, :].cpu().float()
        return LatentCache.of_clip(
            latents,
            frame0_latent,
            source_video=source_video,
            vae="cogvideox-5b",
            scaling_factor=LATENT_SCALE,
            frame0_encoding="repeat_padded_49",
        )


class CogVideoXPromptEncoder:
    """The CogVideoX T5 text cache (``cache/cogvideox/text/``)."""

    text_token_std = TEXT_TOKEN_STD

    def load(self, device: str) -> None:
        from transformers import T5EncoderModel, T5Tokenizer

        logger.info(f"loading T5 from {MODEL_ID}")
        tokenizer = T5Tokenizer.from_pretrained(
            MODEL_ID, subfolder="tokenizer", revision=MODEL_REVISION
        )
        text_encoder = T5EncoderModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder", revision=MODEL_REVISION, torch_dtype=torch.float16
        )
        num_added = tokenizer.add_tokens(trajectory_token_strings())
        text_encoder.resize_token_embeddings(len(tokenizer))
        logger.info(
            f"registered {num_added} <traj_i> tokens; vocab={len(tokenizer)}, "
            f"is_fast={tokenizer.is_fast}"
        )
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(device).eval()
        self.device = device

    def tokenize(self, prompt: str, stem: str) -> TokenizedPrompt:
        encoding = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=TEXT_SEQ_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)  # (1, TEXT_SEQ_LENGTH)
        return TokenizedPrompt(
            input_ids=input_ids,
            token_ids=input_ids[0].tolist(),
            offsets=[tuple(o) for o in encoding["offset_mapping"][0].tolist()],
        )

    def trajectory_token_id(self, index: int) -> int:
        return self.tokenizer.convert_tokens_to_ids(f"<traj_{index}>")

    def input_embeddings(self, tokenized: TokenizedPrompt) -> torch.Tensor:
        return self.text_encoder.get_input_embeddings()(tokenized.input_ids)  # fp16

    def encode(self, input_embeds: torch.Tensor, tokenized: TokenizedPrompt) -> torch.Tensor:
        embedding = self.text_encoder(inputs_embeds=input_embeds.half())[0]  # (1, L, 4096)
        return embedding.squeeze(0).cpu().float()

    def extra_fields(self, files: ClipFiles) -> dict[str, Any]:
        return {}

    def report(self, stem: str, payload: dict[str, Any]) -> None:
        token_ids = self.tokenizer(
            payload["prompt"],
            add_special_tokens=True,
            padding="max_length",
            max_length=TEXT_SEQ_LENGTH,
            truncation=True,
        )["input_ids"]
        to_tokens = self.tokenizer.convert_ids_to_tokens
        logger.info(
            f"{stem}: {payload['num_objects']} objects, "
            f"embedding {tuple(payload['prompt_embedding'].shape)}"
        )
        for i, pos in payload["trajectory_token_positions"].items():
            logger.info(f"  <traj_{i}> at {pos}: {to_tokens([token_ids[pos]])[0]!r}")
        for i, pos in payload["noun_token_positions"].items():
            head_noun = payload["object_labels"][i].split()[-1]
            logger.info(
                f"  noun {i} at {pos}: {to_tokens([token_ids[pos]])[0]!r} (head noun {head_noun!r})"
            )
