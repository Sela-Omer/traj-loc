"""On-disk layout of a dataset root (the contract between all stages).

::

    <root>/
      videos/<stem>.mp4             Stage 1: 49 frames of 480x720 (HxW), written at 24 fps
      first_frames/<stem>.png       Stage 1
      trajectories/<stem>.json      Stage 1: per-object positions, boxes, depths
      prompts/<stem>.json           Stage 1: the prompt, and each object's words in it
      captions/<stem>.json          Stage 1: the scene caption (captioned datasets)
      splits.json                   {"train": [stems], "test": [stems]}
      cache/<backbone>/latents/<stem>.pt   Stage 2: VAE latents
      cache/<backbone>/text/<stem>.pt      Stage 2: text-encoder cache

The Stage-1 records are documented in :mod:`trajloc.preprocess.stage1.records`,
the caches in :mod:`trajloc.datasets.caches`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, get_args

# Every clip, of every dataset: what Stage 1 writes and training reads.
CLIP_NUM_FRAMES = 49
CLIP_HEIGHT = 480
CLIP_WIDTH = 720
CLIP_FPS = 24  # the frame rate videos/ are written at (the frames are the source's)

BackboneName = Literal["cogvideox", "wan"]
BACKBONE_NAMES: tuple[BackboneName, ...] = get_args(BackboneName)
CacheKind = Literal["latents", "text"]
CACHE_KINDS: tuple[CacheKind, ...] = get_args(CacheKind)

# The directory names inside a dataset root (relative to the root).
VIDEOS_DIR = "videos"
FIRST_FRAMES_DIR = "first_frames"
TRAJECTORIES_DIR = "trajectories"
PROMPTS_DIR = "prompts"
CAPTIONS_DIR = "captions"
SPLITS_FILE = "splits.json"
CLIP_DIRS = (VIDEOS_DIR, FIRST_FRAMES_DIR, TRAJECTORIES_DIR, PROMPTS_DIR)  # every clip has all four
# The builders' bookkeeping directories, for resuming and debugging (nothing downstream reads them).
OBJECTS_DIR = "objects"  # benchmarks: Stage 1A's tracked objects, before naming
OBJECT_CROPS_DIR = "object_crops"  # benchmarks: the crops Stage 1B names objects from
DONE_DIR = "done"  # a finished video (SA-V) or clip (benchmark naming)
ERRORS_DIR = "errors"  # why a video or clip failed
TIMINGS_DIR = "timings"  # seconds per stage
TMP_DIR = "tmp"  # benchmarks: SAM 2's per-clip frames


def cache_subdir(backbone: BackboneName, kind: CacheKind) -> str:
    """``cache/<backbone>/<kind>``, relative to a dataset root."""
    if backbone not in BACKBONE_NAMES:
        raise ValueError(f"unknown backbone {backbone!r}; expected one of {BACKBONE_NAMES}")
    if kind not in CACHE_KINDS:
        raise ValueError(f"unknown cache kind {kind!r}; expected one of {CACHE_KINDS}")
    return f"cache/{backbone}/{kind}"


class ClipFiles(NamedTuple):
    """A clip's Stage-1 files."""

    video: Path  # videos/<stem>.mp4
    trajectories: Path  # trajectories/<stem>.json
    prompts: Path  # prompts/<stem>.json
    first_frame: Path  # first_frames/<stem>.png


@dataclass(frozen=True)
class DatasetRoot:
    """The paths inside a dataset root (see the module docstring)."""

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    @property
    def videos_dir(self) -> Path:
        """``videos/``."""
        return self.path / VIDEOS_DIR

    @property
    def first_frames_dir(self) -> Path:
        """``first_frames/``."""
        return self.path / FIRST_FRAMES_DIR

    @property
    def trajectories_dir(self) -> Path:
        """``trajectories/``."""
        return self.path / TRAJECTORIES_DIR

    @property
    def prompts_dir(self) -> Path:
        """``prompts/``."""
        return self.path / PROMPTS_DIR

    @property
    def captions_dir(self) -> Path:
        """``captions/``."""
        return self.path / CAPTIONS_DIR

    @property
    def splits_path(self) -> Path:
        """``splits.json``."""
        return self.path / SPLITS_FILE

    def clip_files(self, stem: str) -> ClipFiles:
        """The paths of clip ``stem``'s Stage-1 files."""
        return ClipFiles(
            video=self.videos_dir / f"{stem}.mp4",
            trajectories=self.trajectories_dir / f"{stem}.json",
            prompts=self.prompts_dir / f"{stem}.json",
            first_frame=self.first_frames_dir / f"{stem}.png",
        )

    def stems(self) -> list[str]:
        """The stems with all four Stage-1 files, sorted (so a Stage-1 root
        that is still being written can be read)."""
        videos = {p.stem for p in self.videos_dir.glob("*.mp4")}
        trajectories = {p.stem for p in self.trajectories_dir.glob("*.json")}
        prompts = {p.stem for p in self.prompts_dir.glob("*.json")}
        first_frames = {p.stem for p in self.first_frames_dir.glob("*.png")}
        return sorted(videos & trajectories & prompts & first_frames)

    def latent_cache_dir(self, backbone: BackboneName) -> Path:
        """``cache/<backbone>/latents/``: the backbone's VAE latents."""
        return self.cache_dir(backbone, "latents")

    def text_cache_dir(self, backbone: BackboneName) -> Path:
        """``cache/<backbone>/text/``: the backbone's text-encoder caches."""
        return self.cache_dir(backbone, "text")

    def cache_dir(self, backbone: BackboneName, kind: CacheKind) -> Path:
        """``cache/<backbone>/<kind>/``."""
        return self.path / cache_subdir(backbone, kind)
