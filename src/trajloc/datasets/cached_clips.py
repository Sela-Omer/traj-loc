"""Training clips read from a dataset root's Stage-2 caches.

A sample (:class:`ClipSample`) is one clip: its VAE latents, its text-encoder
cache (prompt embedding, trajectory tensors, token positions) and, for the
loss mask, its per-frame object boxes. Nothing is decoded or augmented at
training time. The Wan caches also carry ``image_embeddings`` and
``i2v_conditioning``, which are passed through (``None`` for CogVideoX).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from trajloc.core.box_mask import BoxLossMask
from trajloc.core.config import BoxMaskSpec
from trajloc.core.latent_grid import VAE_SPATIAL_STRIDE, VAE_TEMPORAL_STRIDE
from trajloc.datasets.caches import LatentCache, TextCache
from trajloc.datasets.layout import BackboneName, DatasetRoot

logger = logging.getLogger(__name__)

# Per frame, the pixel boxes ``(x, y, w, h)`` of the objects visible in it.
FrameBoxes = list[list[tuple[float, float, float, float]]]


@dataclass(frozen=True)
class ClipSample:
    """One training clip (:meth:`CachedClipDataset.__getitem__`); batched by
    :func:`~trajloc.datasets.collate.cached_collate`.

    Attributes:
        latents: ``(C, F, H, W)`` VAE latents.
        frame0_latent: ``(C, H, W)``.
        prompt_embedding: ``(L, D)``.
        trajectory_coords, trajectory_starts, num_objects,
        noun_token_positions, trajectory_token_positions: as cached
            (:class:`~trajloc.datasets.caches.TextCache`).
        box_mask: ``(F, H, W)`` box loss mask.
        image_embeddings, i2v_conditioning: Wan caches only (else ``None``).
        stem, dataset_name, text_cache_path: where it came from.
    """

    latents: torch.Tensor
    frame0_latent: torch.Tensor
    prompt_embedding: torch.Tensor
    trajectory_coords: torch.Tensor
    trajectory_starts: torch.Tensor
    num_objects: int
    noun_token_positions: dict[int, int]
    trajectory_token_positions: dict[int, int]
    box_mask: torch.Tensor
    image_embeddings: torch.Tensor | None
    i2v_conditioning: torch.Tensor | None
    stem: str
    dataset_name: str
    text_cache_path: str


class CachedClipDataset(Dataset[ClipSample]):
    """One split of a dataset root's cached clips, for one backbone.

    Args:
        root: the dataset root: ``splits.json``, ``trajectories/`` (for the
            box loss mask) and the backbone's ``cache/<backbone>/{latents,text}/``.
        backbone: whose caches to read.
        dataset_name: label used in logs and in every sample.
        split: ``"train"`` or ``"test"``.
        num_frames: video frames per clip (49).
        video_size: ``(H, W)`` in pixels.
        box_mask: the box mask of the box-masked loss.
    """

    def __init__(
        self,
        root: str | Path,
        backbone: BackboneName,
        *,
        dataset_name: str,
        split: str = "train",
        num_frames: int,
        video_size: tuple[int, int],
        box_mask: BoxMaskSpec,
    ) -> None:
        self.box_mask_style = box_mask.style
        self.root = DatasetRoot(Path(root))
        self.dataset_name = dataset_name
        self.latent_dir = self.root.latent_cache_dir(backbone)
        self.text_cache_dir = self.root.text_cache_dir(backbone)
        self.num_frames = num_frames
        self.video_size = video_size

        H, W = video_size
        self.latent_height = H // VAE_SPATIAL_STRIDE
        self.latent_width = W // VAE_SPATIAL_STRIDE

        splits_path = self.root.splits_path
        with open(splits_path) as f:
            splits = json.load(f)
        if split not in splits:
            raise KeyError(f"split={split!r} not in {splits_path} (have: {list(splits)})")
        self.split_ids = set(splits[split])

        self.box_mask_builder = BoxLossMask(
            latent_size=(self.latent_height, self.latent_width),
            temporal_padding_frames=box_mask.padding_frames,
            temporal_compression=VAE_TEMPORAL_STRIDE,
        )

        self.samples = self._build_sample_index()
        logger.info(f"CachedClipDataset[{dataset_name}, {split}]: {len(self.samples)} samples")

    # ------------------------------------------------------------- index build

    def _build_sample_index(self) -> list[dict[str, str]]:
        """Clips of the split that have both a cache and a latent file, in
        sorted cache order."""
        if not self.text_cache_dir.exists():
            raise FileNotFoundError(f"text cache dir not found: {self.text_cache_dir}")
        if not self.latent_dir.exists():
            raise FileNotFoundError(f"Latent dir not found: {self.latent_dir}")

        latent_files = sorted(self.latent_dir.glob("*.pt"))
        latent_by_stem = {f.stem: f for f in latent_files}

        samples: list[dict[str, str]] = []
        no_latent = 0
        all_cache_stems: set[str] = set()

        for cache_file in sorted(self.text_cache_dir.glob("*.pt")):
            stem = cache_file.stem
            all_cache_stems.add(stem)
            if stem not in self.split_ids:
                continue

            latent_path = latent_by_stem.get(stem)
            if latent_path is None:
                no_latent += 1
                continue

            samples.append(
                {
                    "stem": stem,
                    "text_cache_path": str(cache_file),
                    "latent_path": str(latent_path),
                }
            )

        if no_latent:
            logger.warning(
                f"CachedClipDataset[{self.dataset_name}]: {no_latent} cache files "
                f"had no matching latent (skipped)"
            )

        if not samples:
            raise RuntimeError(
                f"CachedClipDataset[{self.dataset_name}]: no samples for split. "
                f"cache_dir={self.text_cache_dir} (cache stems: "
                f"{sorted(all_cache_stems)[:3]}...), "
                f"split has {len(self.split_ids)} IDs, "
                f"overlap={len(all_cache_stems & self.split_ids)}, "
                f"latents={len(latent_by_stem)}"
            )

        return samples

    # ------------------------------------------------------------- box masks

    def _frame_boxes(self, stem: str, num_objects: int) -> FrameBoxes:
        """Per-frame pixel boxes of the first ``num_objects`` objects, from
        ``trajectories/<stem>.json`` (boxes ``[x, y, w, h]`` in [0, 1], or
        ``null``, per frame)."""
        trajectories_path = self.root.clip_files(stem).trajectories
        if not trajectories_path.exists():
            raise FileNotFoundError(
                f"trajectories file (for the box mask) not found: {trajectories_path}"
            )

        with open(trajectories_path) as f:
            data = json.load(f)
        H, W = self.video_size
        T = self.num_frames
        boxes_per_object = []
        for n in range(num_objects):  # every object has a box (or null) per frame
            object_boxes = data["objects"][str(n)]["boxes"]
            if len(object_boxes) != T:
                raise ValueError(
                    f"{trajectories_path}: object {n} has {len(object_boxes)} boxes, expected {T}"
                )
            boxes_per_object.append(object_boxes)

        frame_boxes: FrameBoxes = []
        for t in range(T):
            this_frame: list[tuple[float, float, float, float]] = []
            for object_boxes in boxes_per_object:
                if object_boxes[t] is not None:
                    x, y, w, h = object_boxes[t]
                    this_frame.append((x * W, y * H, w * W, h * H))
            frame_boxes.append(this_frame)
        return frame_boxes

    # -------------------------------------------------------------- sample I/O

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ClipSample:
        info = self.samples[idx]

        cache = TextCache.load(info["text_cache_path"])
        num_objects = cache.num_objects
        if num_objects < 1:
            raise ValueError(f"{info['text_cache_path']} has no objects")

        latent_cache = LatentCache.load(info["latent_path"])
        latents = latent_cache.latents

        if latents.shape[2] != self.latent_height or latents.shape[3] != self.latent_width:
            raise ValueError(
                f"Latent shape mismatch for {info['latent_path']}: "
                f"got {tuple(latents.shape)}, expected (*, *, "
                f"{self.latent_height}, {self.latent_width})"
            )

        box_mask = self.box_mask_builder.mask(
            self.box_mask_style,
            self._frame_boxes(info["stem"], num_objects),
            num_frames=self.num_frames,
            video_size=self.video_size,
        )

        return ClipSample(
            latents=latents,
            frame0_latent=latent_cache.frame0_latent,
            prompt_embedding=cache.prompt_embedding,
            trajectory_coords=cache.trajectory_coords,
            trajectory_starts=cache.trajectory_starts,
            num_objects=num_objects,
            noun_token_positions=cache.noun_token_positions,
            trajectory_token_positions=cache.trajectory_token_positions,
            box_mask=box_mask,
            image_embeddings=cache.image_embeddings,
            i2v_conditioning=cache.i2v_conditioning,
            stem=info["stem"],
            dataset_name=self.dataset_name,
            text_cache_path=info["text_cache_path"],
        )
