#!/usr/bin/env python3
"""Generate Query-Aware Scene Graphs by prompting the VLM on shot frames."""

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llava.conversation import conv_templates  # noqa: E402
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.eval.infer import (  # noqa: E402
    build_query_text,
    compute_frame_relevance,
    detect_shots,
    get_clip_bundle,
    load_pretrained_model,
    read_frame,
    safe_json_parse,
)
from llava.mm_utils import tokenizer_image_token  # noqa: E402

PROMPT_HEADER = (
    "You are analyzing a video (or representative frames) to answer a question.\n"
    "Describe all visually explicit entities, attributes, and relations that are relevant to answering it.\n"
    "Do NOT include speculative or inferred content—only what is clearly visible.\n"
    "Output must be wrapped between <json> and </json> tags.\n"
    "Inside the tags, return a valid, self-contained JSON object with two keys: 'objects'.\n"
    "Follow this schema strictly:\n"
    "<json>\n"
    "{\n"
    '  "objects": [ {"name": <object_name>, "attributes": [<visible attributes>]} ],\n'
    "}\n"
    "</json>\n"
    "Rules:\n"
    "- Include visually explicit entities relevant to the question.\n"
    "- Keep descriptions concise (avoid adjectives not visible in frames).\n"
    "Example:\n"
    "<json>\n"
    '{  \"objects\": [{\"name\": \"man\", \"attributes\": [\"wearing red shirt\", \"smiling\"]}],\n'
    "</json>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce query-aware scene graphs for a video QA dataset using representative shot frames."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="lmms-lab/LLaVA-Video-7B-Qwen2",
        help="HuggingFace repo or local path to the video-capable VLM checkpoint.",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default=None,
        help="Base model path if the checkpoint is a LoRA or delta weight.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava_qwen",
        help="Model name hint passed to the LLaVA loader.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha value when loading adapters.",
    )
    parser.add_argument(
        "--temporal_pooling",
        type=int,
        default=0,
        help="Temporal pooling factor passed to the model config (0 keeps default).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to decode per prompt.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/video_understanding/qwen_gem/DATAS/eval/VideoMME/formatted_dataset_10.json",
        help="Path to the JSON dataset consumed by llava.eval.infer.",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="/home/video_understanding/qwen_gem/DATAS/eval/VideoMME/videos/data",
        help="Directory containing the video files referenced by the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/video_understanding/qwen_gem/output_1_QASG",
        help="Destination directory for per-sample Q-SGM JSON files.",
    )
    parser.add_argument(
        "--shot_threshold",
        type=float,
        default=27.0,
        help="PySceneDetect ContentDetector threshold (higher yields fewer shots).",
    )
    parser.add_argument(
        "--shot_min_len",
        type=int,
        default=15,
        help="Minimum number of frames for a detected scene.",
    )
    parser.add_argument(
        "--max_shots",
        type=int,
        default=16,
        help="Maximum number of representative shots (frames) to feed into the VLM.",
    )
    parser.add_argument(
        "--min_frame_score",
        type=float,
        default=0.0,
        help="Minimum CLIP similarity for a frame to be kept (0 keeps all shots).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optionally restrict the number of samples processed from the dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even if the target JSON already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run the pipeline without writing outputs (for smoke testing).",
    )
    return parser.parse_args()


def build_prompt(question: str, shot_descriptions: List[str]) -> str:
    shots_text = "\n".join(shot_descriptions) if shot_descriptions else "Shots: (single frame provided)."
    prompt = (
        f"{PROMPT_HEADER}\n\n"
        f"Question: {question.strip()}\n"
        f"{shots_text}\n"
        "Remember: respond exclusively with the <json>...</json> block."
    )
    return prompt


def prepare_shots(sample: Dict[str, Any], args: argparse.Namespace, clip_device: torch.device) -> List[Dict[str, Any]]:
    video_path = Path(args.video_root) / sample["video"]
    shots = detect_shots(str(video_path), threshold=args.shot_threshold, min_scene_len=args.shot_min_len)
    if not shots:
        shots = [(0, 0)]

    query_text, _ = build_query_text(sample)
    clip_bundle = get_clip_bundle(clip_device)

    shot_data: List[Dict[str, Any]] = []
    for idx, (fs, fe) in enumerate(shots):
        rep = (fs + fe) // 2
        try:
            frame_rgb = read_frame(str(video_path), rep)
        except RuntimeError:
            continue
        score = compute_frame_relevance(frame_rgb, query_text, clip_bundle, clip_device)
        shot_data.append(
            {
                "frame": frame_rgb,
                "frame_index": rep,
                "range": (fs, fe),
                "score": float(score),
                "shot_id": idx,
            }
        )

    if not shot_data:
        return []

    threshold = args.min_frame_score
    if threshold > 0:
        filtered = [shot for shot in shot_data if shot["score"] >= threshold]
        if filtered:
            shot_data = filtered

    if len(shot_data) > args.max_shots:
        shot_data = shot_data[: args.max_shots]

    return shot_data


def run_sample(
    sample: Dict[str, Any],
    args: argparse.Namespace,
    tokenizer,
    model,
    image_processor,
    model_device,
    model_dtype,
) -> Dict[str, Any]:
    clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shot_data = prepare_shots(sample, args, clip_device)
    if not shot_data:
        raise RuntimeError(f"No frames could be prepared for sample {sample.get('id')}")

    frames_np = np.stack([shot["frame"] for shot in shot_data])
    video_tensor = image_processor.preprocess(frames_np, return_tensors="pt")["pixel_values"]
    video_tensor = video_tensor.to(device=model_device, dtype=model_dtype)

    shot_descriptions = [
        f"Shot {i + 1}: frames {shot['range'][0]}-{shot['range'][1]} (rep {shot['frame_index']}, relevance {shot['score']:.2f})"
        for i, shot in enumerate(shot_data)
    ]

    prompt_instruction = build_prompt(sample.get("question", ""), shot_descriptions)
    prompt_with_token = DEFAULT_IMAGE_TOKEN + prompt_instruction

    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], prompt_with_token)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model_device)

    with torch.inference_mode():
        cont = model.generate(
            input_ids,
            images=[video_tensor],
            modalities=["video"],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=args.max_new_tokens,
            use_cache=False,
        )
    text_output = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

    def extract_json_block(raw_text: str) -> Dict[str, Any]:
        match = re.search(r"<json>(.*?)</json>", raw_text, re.DOTALL | re.IGNORECASE)
        if not match:
            return safe_json_parse(raw_text) or {}
        raw_json = match.group(1).strip()
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            return safe_json_parse(raw_json) or {}

    parsed = extract_json_block(text_output)
    objects = parsed.get("objects", [])
    relations = parsed.get("relations", [])

    return {
        "id": sample.get("id"),
        "video": sample.get("video"),
        "question": sample.get("question"),
        "objects": objects,
        "relations": relations,
        "raw_output": text_output,
        "shots": [
            {
                "shot_id": shot["shot_id"],
                "frame_index": shot["frame_index"],
                "range": shot["range"],
                "relevance": shot["score"],
            }
            for shot in shot_data
        ],
    }


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError(f"Dataset at {data_path} must be a list of samples.")

    if args.max_samples is not None:
        dataset = dataset[: args.max_samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_map: Any = "auto"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    overwrite_cfg = None
    if isinstance(args.temporal_pooling, int) and args.temporal_pooling > 1:
        overwrite_cfg = {"temporal_pooling": args.temporal_pooling}

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=args.model_name,
        lora_alpha=args.lora_alpha,
        torch_dtype="bfloat16",
        device_map=device_map,
        overwrite_config=overwrite_cfg,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    first_param = next(model.parameters())
    model_device = first_param.device
    model_dtype = first_param.dtype

    iterator = tqdm(dataset, desc="Generating Q-SGM", unit="sample")
    written = 0
    for sample in iterator:
        sample_id = sample.get("id")
        outfile = output_dir / f"{sample_id}.json"
        if outfile.exists() and not args.overwrite:
            continue

        try:
            payload = run_sample(
                sample=sample,
                args=args,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                model_device=model_device,
                model_dtype=model_dtype,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed on sample {sample_id}: {exc}")
            continue

        if args.dry_run:
            continue

        with outfile.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        written += 1

    iterator.close()
    if not args.dry_run:
        print(f"[Q-SGM] Wrote {written} files to {output_dir}")


if __name__ == "__main__":
    main()
