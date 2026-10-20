"""Wan config handling: YAML -> :class:`WanModelSettings`, and the Wan weights.

Light on purpose (no DiffSynth import).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from trajloc.core.config import check_config_keys
from trajloc.hub import fetch
from trajloc.paths import project_root

# The released configs' ``model_path``, a local copy of the base model (Stage 2
# reads Wan's VAE, UMT5 and CLIP from it), downloaded there on first use.
WAN_WEIGHTS_DIR = project_root() / "checkpoints/wan2.1-i2v-14b-480p"
WAN_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN_MODEL_REVISION = "6b73f84e66371cdfe870c72acd6826e1d61cf279"  # the release's
DIT_WEIGHTS_GLOB = "diffusion_pytorch_model*.safetensors"  # the DiT's shards in that copy
DIT_INDEX = "diffusion_pytorch_model.safetensors.index.json"  # lists the shards
# The rest of that copy that a complete download holds: the VAE, UMT5 and CLIP
# checkpoints and the two tokenizers (Wan2.1's ``WAN_CONFIGS["i2v-14B"]`` names
# them; Wan2.1 reads the UMT5 tokenizer by its Hub id, the CLIP one from here).
WAN_OTHER_FILES = (
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "google/umt5-xxl/tokenizer.json",
    "xlm-roberta-large/tokenizer.json",
)


def wan_weights_dir(model_path: str | Path) -> Path:
    """``model_path`` (relative to the repository root) as the directory of
    the Wan weights. The released configs' directory (:data:`WAN_WEIGHTS_DIR`)
    is downloaded on first use, about 77 GB, and an interrupted download is
    resumed; any other must already hold the weights (``FileNotFoundError``
    otherwise)."""
    path = Path(model_path)
    if not path.is_absolute():
        path = project_root() / path
    missing = _missing_weights(path)
    if missing and path.resolve() == WAN_WEIGHTS_DIR.resolve():
        fetch(WAN_MODEL_ID, path, revision=WAN_MODEL_REVISION)
        missing = _missing_weights(path)
    if missing:
        raise FileNotFoundError(
            f"incomplete Wan2.1 I2V-14B weights at {path} (no {', '.join(missing[:3])}): "
            f'download them with `hf download {WAN_MODEL_ID} --local-dir "{path}"`'
        )
    return path


def _missing_weights(path: Path) -> list[str]:
    """The files of the Wan weights that ``path`` lacks: every DiT shard its
    index lists, and :data:`WAN_OTHER_FILES`."""
    index = path / DIT_INDEX
    if not index.is_file():
        return [DIT_INDEX]
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    return [f for f in (*shards, *WAN_OTHER_FILES) if not (path / f).is_file()]


# The settings a Wan config states (all of them, see configs/wan.yaml): the
# model's (read by :meth:`WanModelSettings.from_config`), the training loop's
# (read by ``training.WanTrainingSettings.from_config``) and the ``trajloc``
# section.
MODEL_CONFIG_KEYS = frozenset({"model_path", "target_modules", "rank", "gradient_checkpointing"})
TRAINING_CONFIG_KEYS = frozenset(
    {
        "seed",
        "train_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
        "max_grad_norm",
        "checkpointing_steps",
        "num_workers",
        "pin_memory",
    }
)
CONFIG_KEYS = MODEL_CONFIG_KEYS | TRAINING_CONFIG_KEYS | {"trajloc"}


@dataclass(frozen=True)
class WanModelSettings:
    """How :class:`~trajloc.backbones.wan.model.WanTrajLocModule` builds its
    pipeline: the backbone-side settings of a Wan config (TrajLoc's own are
    its ``trajloc`` section, :class:`~trajloc.core.config.TrajLocConfig`).

    Attributes:
        model_path: the config's ``model_path``: the directory of the Wan
            weights (:func:`wan_weights_dir`).
        lora_target_modules: the DiT projections that get LoRA,
            comma-separated (config ``target_modules``).
        lora_rank: config ``rank``.
        use_gradient_checkpointing: config ``gradient_checkpointing``.
        device: where the pipeline is loaded.
    """

    model_path: str
    lora_target_modules: str
    lora_rank: int
    use_gradient_checkpointing: bool
    device: str = "cpu"

    task: ClassVar[str] = "sft:train"  # DiffSynth's training task
    lora_base_model: ClassVar[str] = "dit"  # the pipeline model LoRA is added to

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> WanModelSettings:
        """The settings of a Wan YAML config. Raises ``ValueError`` unless the
        config states exactly :data:`CONFIG_KEYS`."""
        check_config_keys(config, CONFIG_KEYS, "Wan")
        return cls(
            model_path=config["model_path"],
            lora_target_modules=",".join(config["target_modules"]),
            lora_rank=config["rank"],
            use_gradient_checkpointing=config["gradient_checkpointing"],
        )
