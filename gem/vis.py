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
    hm_verb: np.ndarray,
    hm_object: np.ndarray,
    hm_action: np.ndarray,
) -> None:
    weights = {"verb": W_VERB, "object": W_OBJECT, "action": W_ACTION}
    os.makedirs(out_dir, exist_ok=True)
    num_frames = frames_bgr.shape[0]
    peak_log = {"verb": [], "object": [], "action": [], "final": []}

    for frame_index in range(num_frames):
        heatmaps = {"verb": hm_verb, "object": hm_object, "action": hm_action}
        frame_peaks = {}
        for key, heatmap_stack in heatmaps.items():
            overlay = upsample_and_overlay(frames_bgr[frame_index], heatmap_stack[frame_index])
            resized_heatmap = cv2.resize(
                heatmap_stack[frame_index],
                (frames_bgr[frame_index].shape[1], frames_bgr[frame_index].shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            x_coord, y_coord = peak_coordinate(resized_heatmap)
            cv2.circle(overlay, (x_coord, y_coord), 6, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(out_dir, f"heatmap_{key}_f{frame_index:03d}.png"), overlay)
            peak_log[key].append({"frame": frame_index, "x": x_coord, "y": y_coord})
            frame_peaks[key] = (x_coord, y_coord)

        final_x = sum(frame_peaks[key][0] * weights[key] for key in ["verb", "object", "action"])
        final_y = sum(frame_peaks[key][1] * weights[key] for key in ["verb", "object", "action"])
        peak_log["final"].append({"frame": frame_index, "x": final_x, "y": final_y})

    with open(os.path.join(out_dir, "final_peaks.json"), "w") as fp:
        json.dump(peak_log, fp, indent=2)


def save_meta(out_dir: str, meta: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)
