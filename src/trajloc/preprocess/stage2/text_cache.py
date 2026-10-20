"""Stage-2 text caches: ``prompts/`` + ``trajectories/`` -> ``cache/<backbone>/text/<stem>.pt``.

The step both backbones share (Template Method).

Per clip, :meth:`TextCacheStep.process`:

1. loads the clip's prompt and trajectories, and checks they describe the
   same objects;
2. embeds every trajectory with the frozen trajectory encoder;
3. tokenizes the prompt (the backbone's tokenizer) and maps each
   object's word indices to token positions through the character offsets;
4. checks the token at each ``<traj_i>`` position is ``<traj_i>``;
5. replaces each ``<traj_i>`` input embedding with object ``i``'s trajectory
   embedding and runs the text encoder;
6. assembles the :class:`~trajloc.datasets.caches.TextCache`, with the
   backbone's extra fields.

A backbone supplies its tokenizer and text encoder through its
:class:`~trajloc.backbones.base.PromptEncoder`
(:mod:`trajloc.backbones.cogvideox.stage2`, :mod:`trajloc.backbones.wan.stage2`).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from trajloc.backbones import PromptEncoder, get_backbone
from trajloc.datasets.caches import TextCache
from trajloc.datasets.layout import BackboneName
from trajloc.preprocess.stage1.records import check_identity
from trajloc.preprocess.stage2.clip_inputs import load_json, trajectory_starts, trajectory_tensors
from trajloc.preprocess.stage2.models import (
    TRAJECTORY_ENCODER_CHECKPOINT,
    load_released_trajectory_encoder,
)
from trajloc.preprocess.stage2.step import Payload, Stage2Step
from trajloc.preprocess.stage2.tokens import map_word_indices_to_tokens

logger = logging.getLogger(__name__)


class TextCacheStep(Stage2Step):
    """A backbone's text caches, through its
    :class:`~trajloc.backbones.base.PromptEncoder`.

    Args:
        root: the dataset root.
        backbone: whose prompt encoder.
        dataset_name: recorded in each cache (default: the root directory's name).
        device: torch device string.
        encoder: the backbone's prompt encoder by default.
        trajectory_encoder_path: the trajectory encoder's weights (default:
            ``checkpoints/trajectory_encoder.safetensors``).
    """

    cache_kind = "text"

    def __init__(
        self,
        root: Path,
        backbone: BackboneName,
        *,
        dataset_name: str | None = None,
        device: str = "cuda:0",
        encoder: PromptEncoder | None = None,
        trajectory_encoder_path: Path = TRAJECTORY_ENCODER_CHECKPOINT,
    ) -> None:
        super().__init__(root, backbone, device=device)
        # The root's own name, even for "." (a symlinked root keeps its name).
        name = Path(os.path.abspath(root)).name
        self.dataset_name = dataset_name if dataset_name is not None else name
        self.encoder = encoder or get_backbone(backbone).load_prompt_encoder()
        self.trajectory_encoder_path = trajectory_encoder_path

    def setup(self) -> None:
        # In this order: the encoder adds the <traj_i> tokens (their new
        # embeddings draw from the RNG) and loads its extras, then the
        # trajectory encoder.
        self.encoder.load(self.device)
        self.trajectory_encoder = load_released_trajectory_encoder(
            token_std=self.encoder.text_token_std,
            device=self.device,
            path=self.trajectory_encoder_path,
        )
        logger.info(f"trajectory encoder loaded (token_std={self.encoder.text_token_std})")

    def report(self, stem: str, payload: Payload) -> None:
        self.encoder.report(stem, payload)

    # ---------------------------------------------------------------- template

    @torch.no_grad()
    def process(self, stem: str) -> Payload | None:
        files = self.dataset_root.clip_files(stem)
        traj_data = load_json(files.trajectories)
        prompt_meta = load_json(files.prompts)
        check_identity(prompt_meta, traj_data, stem)

        coords, depths, labels, ids = trajectory_tensors(traj_data)  # (N, 49, 2), (N, 49)
        if coords.shape[0] == 0:
            logger.error(f"  {stem}: the clip has no trajectories")
            return None
        prompt = prompt_meta["prompt"]
        num_objects = coords.shape[0]

        trajectory_embeddings = self.trajectory_encoder(
            coords.to(self.device, dtype=torch.float32).unsqueeze(0),
            depths.to(self.device, dtype=torch.float32).unsqueeze(0),
        )[0]  # (N, D)

        tokenized = self.encoder.tokenize(prompt, stem)
        token_index = map_word_indices_to_tokens(prompt, tokenized.offsets, prompt_meta["objects"])
        trajectory_positions = token_index.trajectory_token_positions
        for i, pos in trajectory_positions.items():
            expected = self.encoder.trajectory_token_id(i)
            if tokenized.token_ids[pos] != expected:
                raise RuntimeError(
                    f"<traj_{i}> verification failed: token at pos {pos} is id "
                    f"{tokenized.token_ids[pos]}, expected {expected}. prompt: {prompt!r}"
                )

        input_embeds = self.encoder.input_embeddings(tokenized).clone()
        for i in range(num_objects):
            if i in trajectory_positions:
                input_embeds[0, trajectory_positions[i], :] = trajectory_embeddings[i, :].to(
                    input_embeds.dtype
                )
        prompt_embedding = self.encoder.encode(input_embeds, tokenized)

        return TextCache(
            prompt_embedding=prompt_embedding,
            trajectory_token_positions=trajectory_positions,
            noun_token_positions=token_index.noun_token_positions,
            trajectory_coords=coords,
            trajectory_depths=depths,
            trajectory_starts=trajectory_starts(coords),
            num_objects=num_objects,
            prompt=prompt,
            stem=stem,
            dataset_name=self.dataset_name,
            object_labels=labels,
            object_ids=ids,
            **self.encoder.extra_fields(files),
        ).to_payload()
