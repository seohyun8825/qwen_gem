import json
import os
from typing import Any, Dict, List

import numpy as np
from decord import VideoReader, cpu


def load_dataset_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as fp:
        return json.load(fp)


def read_video_frames(video_path: str, max_frames: int = 16, force_uniform: bool = True) -> np.ndarray:
    reader = VideoReader(video_path, ctx=cpu(0))
    total = len(reader)
    if force_uniform:
        indices = np.linspace(0, total - 1, max_frames, dtype=int)
    else:
        step = max(1, total // max_frames)
        indices = np.arange(0, total, step)[:max_frames]
    return reader.get_batch(indices).asnumpy()


def bgr(frames_rgb: np.ndarray) -> np.ndarray:
    return frames_rgb[..., ::-1].copy()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
