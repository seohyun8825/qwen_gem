from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import ALPHA


def token_sim_heatmap(tokens: torch.Tensor, embedding: torch.Tensor, has_cls: bool, grid: Tuple[int, int]) -> torch.Tensor:
    embedding = embedding / (embedding.norm() + 1e-6)
    temporal, num_tokens, _ = tokens.shape
    if has_cls:
        patches = tokens[:, 1:, :]
    else:
        patches = tokens
    patches = F.normalize(patches, dim=-1)
    similarities = torch.einsum("tnd,d->tn", patches, embedding)
    height, width = grid
    return similarities.view(temporal, height, width)


def upsample_and_overlay(
    frame_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = ALPHA,
    min_val: float | None = None,
    max_val: float | None = None,
) -> np.ndarray:
    height, width, _ = frame_bgr.shape
    if min_val is None or max_val is None or max_val - min_val < 1e-6:
        h_min = heatmap.min()
        h_max = heatmap.max()
        normalized = (heatmap - h_min) / (h_max - h_min + 1e-6)
    else:
        normalized = (heatmap - min_val) / (max_val - min_val + 1e-6)
        normalized = np.clip(normalized, 0.0, 1.0)
    resized = cv2.resize(normalized.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1.0, colored, alpha, 0.0)


def peak_coordinate(heatmap: np.ndarray) -> Tuple[int, int]:
    y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(x_idx), int(y_idx)
