"""
Runner for the Lexical Gap Alignment experiment.
Computes the delta between top-k frame similarities for the original question and
its visually aligned rewrite, saving per-sample logs and an aggregate summary.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity assuming both vectors are L2-normalized."""
    return torch.sum(a.unsqueeze(0) * b, dim=-1)


def _topk_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return 0.0
    k = max(1, min(k, values.size))
    idx = np.argpartition(values, -k)[-k:]
    return float(np.mean(values[idx]))


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

    logs = []
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

        sims_orig = _cosine_sim(emb_orig, frame_embeddings).detach().cpu().numpy()
        sims_visible = _cosine_sim(emb_visible, frame_embeddings).detach().cpu().numpy()

        if args.topk > 0:
            topk = args.topk
        else:
            topk = max(1, int(math.ceil(len(sims_orig) * 0.1)))

        delta = _topk_mean(sims_visible, topk) - _topk_mean(sims_orig, topk)

        log_item = {
            "id": sample_id,
            "video": video_rel,
            "k": topk,
            "delta_topk_mean": float(delta),
            "sims_orig": sims_orig.tolist(),
            "sims_visible": sims_visible.tolist(),
        }
        logs.append(log_item)

        per_sample_path = os.path.join(args.results_dir, f"{sample_id:06d}.json")
        with open(per_sample_path, "w", encoding="utf-8") as fp:
            json.dump(log_item, fp, indent=2)

    deltas = [item["delta_topk_mean"] for item in logs]
    summary = {
        "num_samples": len(logs),
        "mean_delta_topk_mean": float(np.mean(deltas)) if deltas else 0.0,
        "positive_delta_ratio": float(np.mean([delta > 0 for delta in deltas])) if deltas else 0.0,
        "args": vars(args),
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print("[LexAlign] summary:", summary)


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
    parser.add_argument("--topk", type=int, default=0, help="If 0, use ceil(T*0.1).")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means evaluate all samples.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
