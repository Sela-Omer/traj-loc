"""Qwen2.5-VL, shared by the Stage-1 steps that use it (object naming and
scene captions).

The model is loaded by its hub id, at a pinned revision (``QWEN_REVISION``).
``qwen_vl_utils`` comes from the pinned Wan2.1 checkout (``wan/utils``).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from trajloc.paths import add_to_path, project_root

logger = logging.getLogger(__name__)

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"  # the hub revision Stage 1 uses


def prepare_qwen_utils() -> None:
    """Put ``external/Wan2.1/wan/utils`` (``qwen_vl_utils``) on ``sys.path``."""
    wan_utils = project_root() / "external" / "Wan2.1" / "wan" / "utils"
    if str(wan_utils) not in sys.path:  # raises, with the fix, if the checkout is missing
        add_to_path("external/Wan2.1/wan/utils")


def load_qwen(model_path: str = QWEN_MODEL_ID, device: str = "cuda:0") -> tuple[Any, Any]:
    """``(model, processor)``: Qwen2.5-VL in bfloat16 on ``device``, in eval mode."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    logger.info(f"loading Qwen2.5-VL from {model_path} on {device}...")
    t0 = time.time()
    revision = QWEN_REVISION if model_path == QWEN_MODEL_ID else None  # a local path: as is
    processor = AutoProcessor.from_pretrained(model_path, revision=revision)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    logger.info(f"loaded in {time.time() - t0:.1f}s")
    return model, processor


class QwenVL:
    """Qwen2.5-VL with greedy decoding: the two ways Stage 1 asks it things.

    Args:
        model_path: hub id or local path.
        device: torch device string.
    """

    def __init__(self, model_path: str = QWEN_MODEL_ID, device: str = "cuda:0") -> None:
        prepare_qwen_utils()
        self.model_path = model_path
        self.device = device
        self.model, self.processor = load_qwen(model_path, device)

    def ask_image(self, image: Image.Image, prompt: str, *, max_new_tokens: int) -> str:
        """The answer to ``prompt`` about a PIL ``image``, stripped."""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def ask_video(
        self, video_path: Path, prompt: str, *, nframes: int, max_pixels: int, max_new_tokens: int
    ) -> str:
        """The answer to ``prompt`` about ``nframes`` frames of an mp4 (each
        at most ``max_pixels`` pixels), undecorated."""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "max_pixels": max_pixels,
                        "nframes": nframes,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # The pinned Wan2.1 checkout's qwen_vl_utils (prepare_qwen_utils): it
        # returns no video kwargs.
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs: dict[str, Any] = {}
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0]
