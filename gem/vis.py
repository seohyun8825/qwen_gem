import json
import os
from typing import Dict, List, Sequence

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
    score_log: Dict[str, List[Dict[str, float]]] = {key: [] for key in heatmaps.keys()}
    if weights:
        peak_log["final"] = []
        score_log["final"] = []

    save_raw = os.environ.get("GEM_SAVE_RAW_HEATMAP", "").lower() in {"1", "true", "yes"}
    raw_dir = None
    if save_raw:
        raw_dir = os.path.join(out_dir, "raw_arrays")
        os.makedirs(raw_dir, exist_ok=True)

    for frame_index in range(num_frames):
        frame_peaks = {}
        frame_scores = {}
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
            frame_score = float(heatmap_stack[frame_index].max())
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
            score_log[key].append({"frame": frame_index, "score": frame_score})
            frame_peaks[key] = (x_coord, y_coord)
            frame_scores[key] = frame_score

        if weights:
            valid_keys = [k for k in weights.keys() if k in frame_peaks]
            final_x = sum(frame_peaks[k][0] * weights[k] for k in valid_keys)
            final_y = sum(frame_peaks[k][1] * weights[k] for k in valid_keys)
            peak_log["final"].append({"frame": frame_index, "x": final_x, "y": final_y})
            final_score = sum(frame_scores[k] * weights[k] for k in valid_keys)
            score_log["final"].append({"frame": frame_index, "score": final_score})

    with open(os.path.join(out_dir, "final_peaks.json"), "w") as fp:
        json.dump(peak_log, fp, indent=2)
    with open(os.path.join(out_dir, "frame_scores.json"), "w") as fp:
        json.dump(score_log, fp, indent=2)

    if labels:
        for key, heatmap_stack in heatmaps.items():
            safe_key = str(key).replace(" ", "_")
            prompt_text = labels.get(key, key)
            scores = [entry["score"] for entry in score_log[key]]
            save_relevance_strip(
                out_path=os.path.join(out_dir, f"{safe_key}_relevance.png"),
                frames_bgr=frames_bgr,
                scores=scores,
                prompt=prompt_text,
            )


def save_meta(out_dir: str, meta: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)


def _wrap_text(text: str, max_line_width: int = 18) -> List[str]:
    if not text:
        return [""]
    words = text.split()
    if not words:
        return [text]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_line_width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def save_relevance_strip(
    out_path: str,
    frames_bgr: np.ndarray,
    scores: Sequence[float],
    prompt: str,
    *,
    top_k: int = 3,
    target_frame_height: int = 120,
    label_panel_width: int = 200,
    text_panel_height: int = 32,
) -> None:
    if frames_bgr.ndim != 4:
        raise ValueError(f"Expected frames to have shape [T, H, W, C], got {frames_bgr.shape}")
    num_frames = frames_bgr.shape[0]
    if num_frames == 0:
        return
    if len(scores) != num_frames:
        raise ValueError("scores length must match number of frames")

    source_height, source_width = frames_bgr[0].shape[:2]
    if source_height == 0 or source_width == 0:
        raise ValueError("Invalid frame dimensions")
    aspect_ratio = source_width / source_height
    target_frame_width = max(48, int(round(target_frame_height * aspect_ratio)))

    panel_height = target_frame_height + text_panel_height
    label_panel = np.full((panel_height, label_panel_width, 3), 255, dtype=np.uint8)
    prompt_lines = _wrap_text(prompt, max_line_width=18)
    y_cursor = 18
    for line in prompt_lines:
        cv2.putText(
            label_panel,
            line,
            (10, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y_cursor += 18

    score_array = np.array(scores, dtype=float)
    sorted_indices = np.argsort(score_array)[::-1]
    top_indices = set(sorted_indices[: min(top_k, len(sorted_indices))])

    panels: List[np.ndarray] = [label_panel]
    for idx in range(num_frames):
        frame = frames_bgr[idx]
        resized = cv2.resize(frame, (target_frame_width, target_frame_height), interpolation=cv2.INTER_AREA)
        panel = np.full((panel_height, target_frame_width, 3), 255, dtype=np.uint8)
        panel[:target_frame_height] = resized
        score_text = f"{score_array[idx]:.3f}"
        cv2.putText(
            panel,
            score_text,
            (6, target_frame_height + text_panel_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        if idx in top_indices:
            cv2.rectangle(panel, (1, 1), (target_frame_width - 2, target_frame_height - 2), (0, 0, 255), 2)
        panels.append(panel)

    strip = cv2.hconcat(panels)
    cv2.imwrite(out_path, strip)
