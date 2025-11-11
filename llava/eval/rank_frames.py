import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
import open_clip

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


def detect_shots(video_path: str, threshold: float = 27.0, min_scene_len: int = 15) -> List[Tuple[int, int]]:
    vm = VideoManager([video_path])
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    vm.start()
    sm.detect_scenes(frame_source=vm)
    scene_list = sm.get_scene_list()
    vm.release()
    shots: List[Tuple[int, int]] = []
    for start_time, end_time in scene_list:
        fs, fe = start_time.get_frames(), end_time.get_frames()
        if fe > fs:
            shots.append((fs, fe))
    return shots


def candidate_frames(video_path: str, shot_based: bool, uniform_fps: float) -> Tuple[List[int], np.ndarray, float]:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 30.0
    if shot_based:
        shots = detect_shots(video_path)
        if not shots:
            shots = [(0, total)]
        indices = [(fs + fe) // 2 for fs, fe in shots]
    else:
        step = max(1, int(round(fps / max(uniform_fps, 1e-6))))
        indices = list(range(0, total, step))
    if not indices:
        indices = [total // 2 if total else 0]
    frames = vr.get_batch(indices).asnumpy()
    return indices, frames, fps


def greedy_pick(
    sim_matrix: np.ndarray,
    min_gap_sec: float,
    frame_indices: List[int],
    fps: float,
    per_query_topk: int,
) -> List[Dict[str, Any]]:
    q_count, n_frames = sim_matrix.shape
    picks: List[Dict[str, Any]] = []
    min_gap_frames = int(round(min_gap_sec * max(fps, 1e-6)))

    def overlaps(idx: int) -> bool:
        for chosen in picks:
            if abs(frame_indices[idx] - chosen["frame_index"]) <= min_gap_frames:
                return True
        return False

    for q_idx in range(q_count):
        order = np.argsort(-sim_matrix[q_idx])
        taken = 0
        for cand in order:
            if overlaps(cand):
                continue
            picks.append(
                {
                    "query_rank": q_idx + 1,
                    "candidate_index": int(cand),
                    "frame_index": int(frame_indices[cand]),
                    "similarity": float(sim_matrix[q_idx, cand]),
                    "timestamp_s": float(frame_indices[cand] / max(fps, 1e-6)),
                }
            )
            taken += 1
            if taken >= per_query_topk:
                break
    return picks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--queries_dir", type=str, required=True)
    parser.add_argument("--selected_dir", type=str, required=True)
    parser.add_argument("--uniform_fps", type=float, default=1.0)
    parser.add_argument("--shot_based", action="store_true")
    parser.add_argument("--per_query_topk", type=int, default=1)
    parser.add_argument("--min_gap_sec", type=float, default=0.5)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.selected_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[rank_frames] total samples: {len(data)}")
    for sample in tqdm(data, desc="rank_frames"):
        sample_id = sample["id"]
        video_rel = sample["video"]

        query_json = os.path.join(args.queries_dir, f"{sample_id}_queries.json")
        out_json = os.path.join(args.selected_dir, f"{sample_id}_selected.json")

        if (not args.no_cache) and os.path.exists(out_json):
            continue
        if not os.path.exists(query_json):
            print(f"[WARN] missing queries: {query_json}")
            continue

        with open(query_json, "r", encoding="utf-8") as f:
            query_data = json.load(f)
        queries = query_data.get("queries") or []
        if not queries:
            print(f"[WARN] empty queries for {sample_id}")
            continue

        video_path = os.path.join(args.video_root, video_rel)
        frame_indices, frames, fps = candidate_frames(video_path, args.shot_based, args.uniform_fps)

        with torch.inference_mode():
            img_feats = []
            batch_size = 64
            for idx in range(0, len(frames), batch_size):
                chunk = frames[idx : idx + batch_size]
                images = torch.stack([preprocess(Image.fromarray(arr)) for arr in chunk]).to(device)
                feats = clip_model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                img_feats.append(feats)
            img_feats = torch.cat(img_feats, dim=0)

            texts = [entry["text"] for entry in queries]
            text_tokens = tokenizer(texts).to(device)
            text_feats = clip_model.encode_text(text_tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            sims = (text_feats @ img_feats.T).float().cpu().numpy()

        picks = greedy_pick(sims, args.min_gap_sec, frame_indices, fps, args.per_query_topk)
        candidates = [
            {"frame_index": int(idx), "timestamp_s": float(idx / max(fps, 1e-6))}
            for idx in frame_indices
        ]
        result = {
            "id": sample_id,
            "video": video_rel,
            "fps": float(fps),
            "queries": queries,
            "candidates": candidates,
            "selected": picks,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
