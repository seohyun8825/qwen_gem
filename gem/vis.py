import json
import os
from typing import Dict, List

import cv2
import numpy as np

from .config import W_ACTION, W_OBJECT, W_VERB
from .heatmap import peak_coordinate, upsample_and_overlay


def save_heatmaps_and_peaks(
    out_dir: str,
    frames_bgr: np.ndarray,
    heatmaps: Dict[str, np.ndarray],
    weights: Dict[str, float] | None = None,
    labels: Dict[str, str] | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    num_frames = frames_bgr.shape[0]
    if weights is None and set(heatmaps.keys()) == {"verb", "object", "action"}:
        weights = {"verb": W_VERB, "object": W_OBJECT, "action": W_ACTION}

    peak_log: Dict[str, List[Dict[str, float]]] = {key: [] for key in heatmaps.keys()}
    if weights:
        peak_log["final"] = []

    save_raw = os.environ.get("GEM_SAVE_RAW_HEATMAP", "").lower() in {"1", "true", "yes"}
    raw_dir = None
    if save_raw:
        raw_dir = os.path.join(out_dir, "raw_arrays")
        os.makedirs(raw_dir, exist_ok=True)

    for frame_index in range(num_frames):
        frame_peaks = {}
        for key, heatmap_stack in heatmaps.items():
            safe_key = str(key).replace(" ", "_")
            if save_raw and frame_index == 0:
                np.save(os.path.join(raw_dir, f"{safe_key}.npy"), heatmap_stack)

            stack_min = None
            stack_max = None
            if os.environ.get("GEM_STACK_NORMALIZE", "").lower() in {"1", "true", "yes"}:
                stack_min = float(heatmap_stack.min())
                stack_max = float(heatmap_stack.max())

            overlay = upsample_and_overlay(
                frames_bgr[frame_index],
                heatmap_stack[frame_index],
                min_val=stack_min,
                max_val=stack_max,
            )
            resized_heatmap = cv2.resize(
                heatmap_stack[frame_index],
                (frames_bgr[frame_index].shape[1], frames_bgr[frame_index].shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            x_coord, y_coord = peak_coordinate(resized_heatmap)
            cv2.circle(overlay, (x_coord, y_coord), 6, (255, 255, 255), 2)
            label_text = labels.get(key) if labels else key
            if label_text:
                cv2.putText(
                    overlay,
                    label_text,
                    (10, overlay.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imwrite(os.path.join(out_dir, f"{safe_key}_f{frame_index:03d}.png"), overlay)
            peak_log[key].append({"frame": frame_index, "x": x_coord, "y": y_coord})
            frame_peaks[key] = (x_coord, y_coord)

        if weights:
            valid_keys = [k for k in weights.keys() if k in frame_peaks]
            final_x = sum(frame_peaks[k][0] * weights[k] for k in valid_keys)
            final_y = sum(frame_peaks[k][1] * weights[k] for k in valid_keys)
            peak_log["final"].append({"frame": frame_index, "x": final_x, "y": final_y})

    with open(os.path.join(out_dir, "final_peaks.json"), "w") as fp:
        json.dump(peak_log, fp, indent=2)


def save_meta(out_dir: str, meta: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)
