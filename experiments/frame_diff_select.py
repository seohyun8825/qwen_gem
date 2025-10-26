"""
Runner for the frame-diff based frame selection experiment.
For each sample, compute similarity trajectories for the original and visible
questions, derive |Δs_t|, and export candidate frame indices for downstream QA.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llava.model.builder import load_pretrained_model

from gem.config import DEFAULT_MAX_FRAMES, DEVICE
from gem.io_utils import load_dataset_json
from gem.text_encoder import TextEncoder

from experiments._emb_utils import frame_global_embeddings, get_fused_tokens, text_embeddings


def _argtopk(values: np.ndarray, k: int) -> List[int]:
    if values.size == 0:
        return []
    k = max(1, min(k, values.size))
    idx = np.argpartition(values, -k)[-k:]
    sorted_idx = idx[np.argsort(-values[idx])]
    return sorted_idx.tolist()


def _uniform_indices(length: int, k: int) -> List[int]:
    if length == 0:
        return []
    if k >= length:
        return list(range(length))
    positions = np.linspace(0, length - 1, num=k)
    return np.round(positions).astype(int).tolist()


def _build_id_index(data: List[Dict]) -> Dict[int, Dict]:
    table: Dict[int, Dict] = {}
    for sample in data:
        try:
            key = int(sample["id"])
        except Exception as exc:
            raise ValueError(f"Sample missing integer id field: {sample}") from exc
        table[key] = sample
    return table


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.results_dir, exist_ok=True)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=args.model_name,
        torch_dtype="bfloat16" if torch.cuda.is_available() else "float32",
        device_map="auto",
    )
    text_encoder = TextEncoder(model=model, tokenizer=tokenizer, device=DEVICE)

    data_orig = load_dataset_json(args.data_path_orig)
    data_visible = load_dataset_json(args.data_path_visible)
    visible_by_id = _build_id_index(data_visible)

    selections = []
    max_samples = args.max_samples if args.max_samples > 0 else len(data_orig)
    for sample in data_orig[:max_samples]:
        sample_id = int(sample["id"])
        visible = visible_by_id.get(sample_id)
        if visible is None:
            continue

        video_rel = sample["video"]
        video_path = os.path.join(args.video_root, video_rel)

        _, fused_tokens, has_cls, _ = get_fused_tokens(
            video_path,
            model,
            image_processor,
            text_encoder=text_encoder,
            max_frames=args.max_frames,
        )
        frame_embeddings = frame_global_embeddings(fused_tokens, has_cls)

        question_orig = sample["question"]
        question_visible = visible["question"]

        emb_orig = text_embeddings([question_orig], text_encoder)[0]
        emb_visible = text_embeddings([question_visible], text_encoder)[0]
        frame_embeddings = frame_embeddings.to(device=emb_orig.device, dtype=emb_orig.dtype)

        sims_orig = torch.sum(emb_orig.unsqueeze(0) * frame_embeddings, dim=-1).detach().cpu().numpy()
        sims_visible = torch.sum(emb_visible.unsqueeze(0) * frame_embeddings, dim=-1).detach().cpu().numpy()
        sims_diff = np.abs(sims_visible - sims_orig)

        frame_count = sims_orig.size
        topn = max(1, min(args.topn, frame_count))

        indices_diff = _argtopk(sims_diff, topn)
        indices_visible = _argtopk(sims_visible, topn)
        indices_uniform = _uniform_indices(frame_count, topn)

        selections.append(
            {
                "id": sample_id,
                "video": video_rel,
                "frame_indices": {
                    "diff": indices_diff,
                    "align": indices_visible,
                    "uniform": indices_uniform,
                },
                "question_orig": question_orig,
                "question_visible": question_visible,
                "candidates": sample.get("candidates", []),
                "answer": sample.get("answer"),
            }
        )

    output_path = os.path.join(args.results_dir, "selection.json")
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(selections, fp, indent=2)
    print(f"[FrameDiffSelect] saved: {output_path} (samples={len(selections)})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_base", default=None)
    parser.add_argument("--model_name", default="llava_qwen")
    parser.add_argument("--data_path_orig", required=True)
    parser.add_argument("--data_path_visible", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--topn", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means evaluate all samples.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
