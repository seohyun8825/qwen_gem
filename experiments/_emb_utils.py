"""
Utilities shared across experimental scripts for extracting fused video tokens and
producing lightweight embeddings from the existing GEM stack.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gem.config import DEFAULT_MAX_FRAMES, DEVICE
from gem.gem_core import gem_whitebox
from gem.io_utils import read_video_frames
from gem.text_encoder import TextEncoder
from gem.vision_introspect import ViTWhitebox


def _locate_mm_projector(model) -> torch.nn.Module | None:
    """Find the multimodal projector inside a LLaVA-style model."""
    visited: set[int] = set()
    queue: list[object] = []

    def enqueue(item: object | None) -> None:
        if item is None:
            return
        identifier = id(item)
        if identifier in visited:
            return
        visited.add(identifier)
        queue.append(item)

    enqueue(model)
    for attr in ("model", "module", "backbone", "language_model"):
        enqueue(getattr(model, attr, None))

    get_model = getattr(model, "get_model", None)
    if callable(get_model):
        try:
            base = get_model()
            enqueue(base)
            for attr in ("model", "module", "backbone"):
                enqueue(getattr(base, attr, None))
        except Exception:
            pass

    for candidate in queue:
        projector = getattr(candidate, "mm_projector", None)
        if projector is not None:
            return projector
    return None


def _apply_mm_projector(model, tokens: torch.Tensor) -> torch.Tensor:
    projector = _locate_mm_projector(model)
    if projector is None:
        return tokens
    param = next(projector.parameters(), None)
    if param is not None and (tokens.device != param.device or tokens.dtype != param.dtype):
        tokens = tokens.to(device=param.device, dtype=param.dtype)
    return projector(tokens)


def _preprocess_frames(image_processor, frames_rgb: np.ndarray) -> torch.Tensor:
    """Mirror the preprocessing pipeline used by cli.py."""
    pil_frames = [Image.fromarray(frame.astype(np.uint8)).convert("RGB") for frame in frames_rgb]
    batch = image_processor.preprocess(pil_frames, return_tensors="pt")
    return batch["pixel_values"]


@torch.no_grad()
def get_fused_tokens(
    video_path: str,
    model,
    image_processor,
    text_encoder: TextEncoder | None = None,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    text_prompts: Sequence[str] | None = None,
) -> tuple[np.ndarray, torch.Tensor, bool, tuple[int, int]]:
    """
    Load frames, extract ViT tokens, optionally run GEM whitebox fusion, and apply
    the multimodal projector so the tokens live in the same space used at inference.
    """
    frames_rgb = read_video_frames(video_path, max_frames=max_frames, force_uniform=True)
    pixel_values = _preprocess_frames(image_processor, frames_rgb).to(DEVICE)

    vit = ViTWhitebox(model, device=DEVICE)
    vit_out = vit.encode_frames(pixel_values)
    layers = vit_out["layers"]
    qkv = vit_out["qkv"]
    has_cls = vit_out["has_cls"]
    grid = vit_out["grid"]
    vit.remove_hooks()

    disable_whitebox = os.environ.get("GEM_DISABLE_WHITEBOX", "").lower() in {"1", "true", "yes"}
    if disable_whitebox:
        fused_tokens = layers[-1]
    else:
        if text_prompts and text_encoder is not None:
            text_embeds = text_encoder.encode(list(text_prompts))
            text_eos = text_embeds[-1]
        else:
            # Fallback to a zero vector to keep the fusion pipeline operational.
            text_eos = torch.zeros(layers[-1].shape[-1], device=DEVICE, dtype=layers[-1].dtype)
        fused_outputs = gem_whitebox(layers, qkv, has_cls, text_eos=text_eos)
        fused_tokens = fused_outputs["O_comb"]

    fused_tokens = _apply_mm_projector(model, fused_tokens).detach()
    return frames_rgb, fused_tokens, has_cls, grid


@torch.no_grad()
def frame_global_embeddings(fused_tokens: torch.Tensor, has_cls: bool) -> torch.Tensor:
    """
    Pool per-token representations into per-frame embeddings using mean pooling
    over spatial tokens (excluding the CLS token when present) and apply L2 norm.
    """
    if has_cls:
        spatial_tokens = fused_tokens[:, 1:, :]
    else:
        spatial_tokens = fused_tokens
    pooled = spatial_tokens.mean(dim=1)
    return F.normalize(pooled, dim=-1)


@torch.no_grad()
def text_embeddings(
    texts: Iterable[str],
    text_encoder: TextEncoder,
    *,
    device: str = DEVICE,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Encode arbitrary text strings using mean pooled token embeddings (excluding PAD/EOS).
    Using TextEncoder.encode() would return only the final token (often EOS), which
    collapses distinct prompts into the same vector. This helper instead pools token
    embeddings to capture sentence-level differences.
    """

    tokenizer = text_encoder.tokenizer
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Move to the embedding layer device for lookup.
    embed_layer = text_encoder.embed_tokens
    target_device = embed_layer.weight.device
    target_dtype = embed_layer.weight.dtype

    input_ids = encoded["input_ids"].to(device=target_device)
    attention_mask = encoded["attention_mask"].to(device=target_device, dtype=torch.float32)

    token_embeds = embed_layer(input_ids).to(dtype=target_dtype)

    if pad_token_id is not None:
        pad_mask = (input_ids != pad_token_id).to(dtype=torch.float32)
        attention_mask = attention_mask * pad_mask
    if eos_token_id is not None:
        eos_mask = (input_ids != eos_token_id).to(dtype=torch.float32)
        attention_mask = attention_mask * eos_mask

    valid_counts = attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (token_embeds * attention_mask.unsqueeze(-1)).sum(dim=1) / valid_counts

    pooled = pooled.to(device=device, dtype=dtype)
    return F.normalize(pooled, dim=-1)


__all__ = [
    "get_fused_tokens",
    "frame_global_embeddings",
    "text_embeddings",
]
