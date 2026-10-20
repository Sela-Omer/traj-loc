"""Wan's Stage-2 encoders: the VAE latents and the UMT5 text caches.

- :class:`WanVideoEncoder`: Wan 2.1's VAE; ``frame0_latent = latents[:, 0]``
  (Wan conditions on frame 0 through its separate I2V input, built with the
  text cache).
- :class:`WanPromptEncoder`: Wan's UMT5 tokenizer (with the ``<traj_i>``
  tokens added, new embeddings ``N(0, 0.02^2)``) and encoder (Wan's T5
  patched to take input embeddings); the cache keeps the unpadded length.
  Also cached, from frame 0: Wan's I2V conditioning and its CLIP image
  embeddings.

The prompt must be a fixed point of Wan's tokenizer cleaning (see
:mod:`trajloc.preprocess.text`): a clip whose wrapper tokens differ from the
raw tokenizer's fails, since the positions would be wrong.

The encoders are the released base model's, from ``WAN_WEIGHTS_DIR`` (the
caches are part of the data, not of a training config).

Stage 2 needs Wan2.1's modules only (not DiffSynth, which the model and
inference import through :mod:`~trajloc.backbones.wan.upstream`), so this
module puts only ``external/Wan2.1`` on the path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import torch

from trajloc.backbones.base import TokenizedPrompt
from trajloc.backbones.wan.config import WAN_WEIGHTS_DIR, wan_weights_dir
from trajloc.backbones.wan.constants import TEXT_TOKEN_STD, WAN_CONFIG_NAME
from trajloc.backbones.wan.text_encoder import install_t5_inputs_embeds_patch
from trajloc.core.latent_grid import VAE_TEMPORAL_STRIDE
from trajloc.core.trajectory_encoder import trajectory_token_strings
from trajloc.datasets.caches import LatentCache
from trajloc.datasets.layout import CLIP_NUM_FRAMES, ClipFiles
from trajloc.paths import add_to_path

logger = logging.getLogger(__name__)

NEW_TOKEN_INIT_STD = 0.02  # the <traj_i> input embeddings start as N(0, 0.02^2)
PAD_TOKEN_ID = 0  # UMT5's padding id


def _wan_config():
    add_to_path("external/Wan2.1")
    from wan.configs import WAN_CONFIGS

    return WAN_CONFIGS[WAN_CONFIG_NAME]


class WanVideoEncoder:
    """Wan2.1 VAE latents (``cache/wan/latents/``)."""

    def load(self, device: str) -> None:
        wan_config = _wan_config()
        from wan.modules.vae import WanVAE

        logger.info("loading the Wan VAE")
        self.vae = WanVAE(
            vae_pth=str(wan_weights_dir(WAN_WEIGHTS_DIR) / wan_config.vae_checkpoint), device=device
        )
        self.device = device

    @torch.no_grad()
    def encode(self, video: torch.Tensor, source_video: str) -> LatentCache:
        latents = self.vae.encode([video.permute(1, 0, 2, 3).to(self.device)])[0].cpu().float()
        return LatentCache.of_clip(
            latents,
            latents[:, 0, :, :].clone(),
            source_video=source_video,
            vae="wan2.1",
            scaling_factor=None,
        )


class WanPromptEncoder:
    """The Wan UMT5 text cache, with Wan's image conditioning (``cache/wan/text/``)."""

    text_token_std = TEXT_TOKEN_STD

    def load(self, device: str) -> None:
        """The text encoder (its new ``<traj_i>`` embeddings draw from the
        RNG), then the VAE and CLIP of the image conditioning."""
        self.device = device
        self._load_text_encoder(device)
        self._load_image_models(device)

    def _load_text_encoder(self, device: str) -> None:
        self.wan_config = _wan_config()
        from wan.modules.t5 import T5EncoderModel as WanT5

        logger.info("loading Wan's UMT5")
        text_encoder = WanT5(
            text_len=self.wan_config.text_len,
            dtype=self.wan_config.t5_dtype,
            device=device,
            checkpoint_path=str(wan_weights_dir(WAN_WEIGHTS_DIR) / self.wan_config.t5_checkpoint),
            tokenizer_path=self.wan_config.t5_tokenizer,
        )
        # Upstream's T5 forward takes only ids; the <traj_i> embeddings are
        # injected before the encoder runs.
        install_t5_inputs_embeds_patch(text_encoder.model)

        tokenizer = text_encoder.tokenizer.tokenizer  # the HF tokenizer inside Wan's wrapper
        num_added = tokenizer.add_tokens(trajectory_token_strings())
        old_vocab = text_encoder.model.token_embedding.num_embeddings
        new_vocab = len(tokenizer)
        if new_vocab > old_vocab:
            old_embedding = text_encoder.model.token_embedding
            new_embedding = torch.nn.Embedding(
                new_vocab,
                old_embedding.embedding_dim,
                device=old_embedding.weight.device,
                dtype=old_embedding.weight.dtype,
            )
            new_embedding.weight.data[:old_vocab] = old_embedding.weight.data
            torch.nn.init.normal_(new_embedding.weight.data[old_vocab:], std=NEW_TOKEN_INIT_STD)
            text_encoder.model.token_embedding = new_embedding
            logger.info(f"resized the token embeddings: {old_vocab} -> {new_vocab}")
        logger.info(
            f"registered {num_added} <traj_i> tokens; vocab={new_vocab}, "
            f"fast tokenizer: {tokenizer.is_fast}"
        )
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def _load_image_models(self, device: str) -> None:
        from wan.modules.clip import CLIPModel
        from wan.modules.vae import WanVAE

        logger.info("loading the Wan VAE")
        self.vae = WanVAE(
            vae_pth=str(wan_weights_dir(WAN_WEIGHTS_DIR) / self.wan_config.vae_checkpoint),
            device=device,
        )
        logger.info("loading CLIP")
        self.image_encoder = CLIPModel(
            dtype=self.wan_config.clip_dtype,
            device=device,
            checkpoint_path=str(wan_weights_dir(WAN_WEIGHTS_DIR) / self.wan_config.clip_checkpoint),
            tokenizer_path=str(wan_weights_dir(WAN_WEIGHTS_DIR) / self.wan_config.clip_tokenizer),
        )

    def tokenize(self, prompt: str, stem: str) -> TokenizedPrompt:
        """Wan's wrapper gives the ids the encoder reads (no fixed-length
        padding: UMT5 takes the true length); the HF tokenizer inside it
        gives the offsets. The two must agree."""
        encoding = self.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
        direct_ids = encoding["input_ids"]
        offsets = [tuple(o) for o in encoding["offset_mapping"]]

        wrapper_ids, wrapper_mask = self.text_encoder.tokenizer(
            [prompt], return_mask=True, add_special_tokens=True
        )
        seq_len = int(wrapper_mask.gt(0).sum(dim=1)[0].item())
        wrapper_list = [int(x) for x in wrapper_ids[0, :seq_len].tolist()]
        if wrapper_list != direct_ids:
            raise RuntimeError(
                f"{stem}: Wan's tokenizer wrapper and the tokenizer inside it disagree: "
                f"wrapper={wrapper_list[:10]}..., underlying={direct_ids[:10]}..."
            )
        return TokenizedPrompt(
            input_ids=wrapper_ids.to(self.device),  # (1, text_len)
            token_ids=direct_ids,
            offsets=offsets,
            attention_mask=wrapper_mask.to(self.device),
            seq_len=seq_len,
        )

    def trajectory_token_id(self, index: int) -> int:
        ids = [
            token_id
            for token_id in self.tokenizer(f"<traj_{index}>", add_special_tokens=False)["input_ids"]
            if token_id != PAD_TOKEN_ID
        ]
        if not ids:
            raise RuntimeError(f"<traj_{index}> tokenized to nothing")
        return ids[0]

    def input_embeddings(self, tokenized: TokenizedPrompt) -> torch.Tensor:
        return self.text_encoder.model.token_embedding(tokenized.input_ids)

    def encode(self, input_embeds: torch.Tensor, tokenized: TokenizedPrompt) -> torch.Tensor:
        embedding = self.text_encoder.model(
            tokenized.input_ids, tokenized.attention_mask, inputs_embeds=input_embeds
        )
        return embedding[0, : tokenized.seq_len, :].cpu().float()  # (seq_len, 4096)

    def extra_fields(self, files: ClipFiles) -> dict[str, Any]:
        """Wan's image conditioning, from frame 0: the I2V input and the CLIP image embeddings."""
        frame0 = read_frame0_tensor(files.video)
        i2v_conditioning = build_i2v_conditioning(self.vae, frame0, self.device)
        image_embeddings = self.image_encoder.visual([frame0[:, None, :, :].to(self.device)])
        return {"i2v_conditioning": i2v_conditioning, "image_embeddings": image_embeddings[0].cpu()}

    def report(self, stem: str, payload: dict[str, Any]) -> None:
        token_ids = self.tokenizer(payload["prompt"], add_special_tokens=True)["input_ids"]
        to_tokens = self.tokenizer.convert_ids_to_tokens
        logger.info(
            f"{stem}: {payload['num_objects']} objects, "
            f"embedding {tuple(payload['prompt_embedding'].shape)}, "
            f"i2v {tuple(payload['i2v_conditioning'].shape)}, "
            f"image embeddings {tuple(payload['image_embeddings'].shape)}"
        )
        for i, pos in payload["trajectory_token_positions"].items():
            logger.info(f"  <traj_{i}> at {pos}: {to_tokens([token_ids[pos]])[0]!r}")


def read_frame0_tensor(video_path: Path) -> torch.Tensor:
    """Frame 0 of an mp4 as ``(3, H, W)`` RGB float32 in [-1, 1]."""
    cap = cv2.VideoCapture(str(video_path))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame 0 of {video_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1.0


@torch.no_grad()
def build_i2v_conditioning(vae: Any, frame0_tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Wan's I2V input: ``[frame 0, 48 black frames]`` through the VAE,
    preceded by the 4 mask channels (frame 0 known, repeated 4x for the
    first latent frame). ``(4 + 16, T_lat, H_lat, W_lat)``."""
    H, W = frame0_tensor.shape[1:]
    i2v_video = torch.cat(
        [
            frame0_tensor.unsqueeze(1),  # (3, 1, H, W)
            torch.zeros(3, CLIP_NUM_FRAMES - 1, H, W),
        ],
        dim=1,
    ).to(device)  # (3, 49, H, W)
    i2v_latent = vae.encode([i2v_video])[0]  # (16, T_lat, H_lat, W_lat)

    H_lat, W_lat = i2v_latent.shape[2], i2v_latent.shape[3]
    mask = torch.zeros(1, CLIP_NUM_FRAMES, H_lat, W_lat, device=device)
    mask[:, 0] = 1.0
    stride = VAE_TEMPORAL_STRIDE  # video frames per latent frame
    mask = torch.cat([mask[:, 0:1].repeat(1, stride, 1, 1), mask[:, 1:]], dim=1)
    mask = mask.view(1, mask.shape[1] // stride, stride, H_lat, W_lat).transpose(1, 2)[0]
    return torch.cat([mask, i2v_latent], dim=0).cpu()
