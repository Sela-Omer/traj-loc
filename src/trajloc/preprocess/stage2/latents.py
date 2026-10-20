"""Stage-2 VAE latents: ``videos/<stem>.mp4`` -> ``cache/<backbone>/latents/<stem>.pt``.

The backbone's :class:`~trajloc.backbones.base.VideoEncoder` does the
encoding (:mod:`trajloc.backbones.cogvideox.stage2`,
:mod:`trajloc.backbones.wan.stage2`).
"""

from __future__ import annotations

import logging
from pathlib import Path

from trajloc.backbones import VideoEncoder, get_backbone
from trajloc.datasets.layout import BackboneName
from trajloc.preprocess.stage2.clip_inputs import read_video_tensor
from trajloc.preprocess.stage2.step import Payload, Stage2Step

logger = logging.getLogger(__name__)


class LatentCacheStep(Stage2Step):
    """A backbone's latent caches: read each clip and encode it."""

    cache_kind = "latents"

    def __init__(
        self,
        root: Path,
        backbone: BackboneName,
        *,
        device: str = "cuda:0",
        encoder: VideoEncoder | None = None,
    ) -> None:
        """``encoder``: the backbone's by default."""
        super().__init__(root, backbone, device=device)
        self.encoder = encoder or get_backbone(backbone).load_video_encoder()

    def setup(self) -> None:
        self.encoder.load(self.device)

    def process(self, stem: str) -> Payload | None:
        video_path = self.dataset_root.clip_files(stem).video
        video = read_video_tensor(video_path)
        if video is None:
            logger.error(f"{stem}: unreadable or short video {video_path}")
            return None
        return self.encoder.encode(video, video_path.name).to_payload()
