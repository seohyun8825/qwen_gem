import os
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import DEVICE


class ClipScorer:
    def __init__(self, model_name: str | None = None, device: str = DEVICE):
        self.model_name = model_name or os.environ.get("GEM_CLIP_MODEL", "openai/clip-vit-large-patch14")
        self.device = device
        self.model: Optional["CLIPModel"] = None
        self.processor: Optional["CLIPProcessor"] = None
        self._dtype: Optional[torch.dtype] = None
        self._load_failed = False
        self._warned = False
        self._loaded_name: Optional[str] = None

    def _ensure_loaded(self) -> bool:
        if self.model is not None and self.processor is not None:
            return True
        if self._load_failed:
            return False
        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except Exception as exc:  # pragma: no cover - import time guard
            self._log_once(f"[GEM][clip] transformers.CLIPModel import failed: {exc}")
            self._load_failed = True
            return False
        try:
            processor = CLIPProcessor.from_pretrained(self.model_name)
            model = CLIPModel.from_pretrained(self.model_name)
        except Exception as exc:  # pragma: no cover - runtime download error
            self._log_once(f"[GEM][clip] loading CLIP model '{self.model_name}' failed: {exc}")
            self._load_failed = True
            return False
        model = model.to(self.device)
        model.eval()
        self.processor = processor
        self.model = model
        self._dtype = next(model.parameters()).dtype
        self._loaded_name = self.model_name
        return True

    def _log_once(self, message: str) -> None:
        if not self._warned:
            print(message)
            self._warned = True

    @torch.no_grad()
    def score_frames(self, frames_rgb: np.ndarray, prompts: Mapping[str, str]) -> Optional[Dict[str, np.ndarray]]:
        if not prompts:
            return None
        if not self._ensure_loaded():
            return None
        assert self.processor is not None
        assert self.model is not None
        if frames_rgb.ndim != 4:
            raise ValueError(f"Expected frames shape [T, H, W, C], got {frames_rgb.shape}")
        num_frames = frames_rgb.shape[0]
        if num_frames == 0:
            return {}

        pil_frames: List[Image.Image] = []
        for idx in range(num_frames):
            frame = frames_rgb[idx]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(frame))

        image_inputs = self.processor(images=pil_frames, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(device=self.device, dtype=self._dtype)
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        image_features = F.normalize(image_features, dim=-1)

        prompt_items = [(key, text) for key, text in prompts.items() if isinstance(text, str) and text.strip()]
        if not prompt_items:
            return None
        prompt_keys = [item[0] for item in prompt_items]
        prompt_texts = [item[1] for item in prompt_items]

        text_inputs = self.processor(text=prompt_texts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
        text_features = self.model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=-1)

        similarities = image_features @ text_features.T
        similarities = similarities.cpu().numpy()

        scores: Dict[str, np.ndarray] = {}
        for idx, key in enumerate(prompt_keys):
            scores[key] = similarities[:, idx]
        return scores


_CLIP_SCORER: Optional[ClipScorer] = None


def get_clip_scorer() -> ClipScorer:
    global _CLIP_SCORER
    if _CLIP_SCORER is None:
        _CLIP_SCORER = ClipScorer()
    return _CLIP_SCORER
