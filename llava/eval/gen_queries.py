import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


def detect_shots(video_path: str, threshold: float = 27.0, min_scene_len: int = 15) -> List[List[int]]:
    """Return [start_frame, end_frame] shot boundaries."""
    vm = VideoManager([video_path])
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    vm.start()
    sm.detect_scenes(frame_source=vm)
    scene_list = sm.get_scene_list()
    vm.release()
    shots = []
    for start_time, end_time in scene_list:
        fs, fe = start_time.get_frames(), end_time.get_frames()
        if fe > fs:
            shots.append([fs, fe])
    return shots


def sample_frames_for_query_stage(
    video_path: str, max_frames: int, shot_based: bool
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Return numpy array of frames (and shot metadata) for lightweight query generation."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    if total == 0:
        raise RuntimeError(f"No frames found in {video_path}")
    avg_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 30.0

    shot_infos: List[Dict[str, Any]] = []
    if shot_based:
        shots = detect_shots(video_path)
        if not shots:
            shots = [[0, total]]
        reps = [(fs + fe) // 2 for fs, fe in shots]
        if len(reps) > max_frames:
            idxs = np.linspace(0, len(reps) - 1, max_frames, dtype=int).tolist()
            reps = [reps[i] for i in idxs]
            shots = [shots[i] for i in idxs]
        indices = reps
        for idx, (fs, fe) in enumerate(shots):
            rep_idx = indices[min(idx, len(indices) - 1)]
            shot_infos.append(
                {
                    "shot": idx + 1,
                    "frame_start": int(fs),
                    "frame_end": int(fe),
                    "timestamp_start": float(fs / max(1e-6, avg_fps)),
                    "timestamp_end": float(fe / max(1e-6, avg_fps)),
                    "rep_frame_index": int(rep_idx),
                }
            )
    else:
        if max_frames >= total:
            indices = list(range(total))
        else:
            indices = np.linspace(0, total - 1, max_frames, dtype=int).tolist()

    frames = vr.get_batch(indices).asnumpy()
    return frames, shot_infos

INSTRUCTION_TMPL = (
    "You will see a few frames from a video and a question about that video.\n"
    "If you are fully confident about the answer based only on what is visible, provide the answer directly.\n"
    "Otherwise, if there is any uncertainty or missing visual evidence, do NOT answer.\n"
    "Instead, propose VISUALLY EXPLICIT retrieval queries describing which additional frames or scenes should be examined to determine the answer.\n"
    "\n"
    "Each query must describe exactly ONE clear visual scene (one decoration type, one action, one state, etc.).\n"
    "Together, these queries should cover every piece of visual evidence required to confidently answer.\n"
    "\n"
    "Make each query a single, concrete, discriminative description including:\n"
    " • spatial layout (left/right/center; top-left/bottom-right),\n"
    " • colors, patterns, or materials,\n"
    " • visible objects, clothing, gestures, and their states or counts,\n"
    " • exact on-screen text in double quotes if present,\n"
    " • relevant background details or framing (close-up, wide, over-shoulder, etc.).\n"
    "\n"
    "Hard constraints:\n"
    " • Do NOT repeat the question text.\n"
    " • Do NOT explain reasoning.\n"
    " • Each query = one sentence (8–25 words), focused on a single discriminative visual moment.\n"
    " • Avoid vague words like “someone”, “a scene”, “some text”.\n"
    " • Rank scenes from most to least useful for answering.\n"
    " • Include at least one query quoting any legible on-screen text if it exists.\n"
    " • Do not mention 'the question', 'the video', or timestamps.\n"
    "\n"
    "Return JSON ONLY in one of these two exact schemas:\n"
    "\n"
    "<answer>\n"
    "{\n"
    '  \"answer\": \"...\"  # only if confident\n'
    "}\n"
    "\n"
    "OR\n"
    "\n"
    "<query_update>\n"
    "{\n"
    '  \"queries\": [{\"rank\": 1, \"text\": \"...\"}, {\"rank\": 2, \"text\": \"...\"}, ...]\n'
    "}\n"
    "\n"
    "No commentary, no extra keys, no trailing commas.\n"
)

def format_question_with_choices(sample: Dict[str, Any]) -> str:
    question = (sample.get("question") or "").strip()
    candidates = sample.get("candidates") or []
    lines: List[str] = []
    if isinstance(candidates, dict):
        iterable = candidates.items()
    else:
        iterable = enumerate(candidates)
    letter_ord = ord("A")
    for key, value in iterable:
        if isinstance(value, dict):
            text = value.get("text") or value.get("answer") or ""
            label = value.get("label")
        else:
            text = str(value)
            label = None
        text = text.strip()
        if not text:
            continue
        if not label:
            if isinstance(key, str):
                label = key
            else:
                label = chr(letter_ord)
                letter_ord += 1
        lines.append(f"{label}. {text}")
    if lines:
        return question + "\nChoices:\n" + "\n".join(lines)
    return question


def build_prompt(sample: Dict[str, Any], shot_infos: List[Dict[str, Any]]) -> str:
    shot_text = ""
    if shot_infos:
        shot_lines = []
        for info in shot_infos:
            shot_lines.append(
                "Shot {shot}: frames {start}-{end} "
                "({t0:.2f}s-{t1:.2f}s) representative frame idx {rep}".format(
                    shot=info["shot"],
                    start=info["frame_start"],
                    end=info["frame_end"],
                    t0=info["timestamp_start"],
                    t1=info["timestamp_end"],
                    rep=info["rep_frame_index"],
                )
            )
        shot_text = "Shots overview:\n" + "\n".join(shot_lines) + "\n"

    question_block = format_question_with_choices(sample)
    return DEFAULT_IMAGE_TOKEN + f"{INSTRUCTION_TMPL}\n\n{shot_text}Question: {question_block}"


def safe_json_parse(text: str) -> Dict[str, Any]:
    payload = text.strip()
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except Exception:
        first, last = payload.find("{"), payload.rfind("}")
        if first != -1 and last != -1 and last > first:
            snippet = payload[first : last + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return {}
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_queries", type=int, default=3)
    parser.add_argument("--query_stage_frames", type=int, default=24)
    parser.add_argument("--shot_based", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name="llava_qwen",
        lora_alpha=None,
        torch_dtype="bfloat16",
        device_map="auto",
        overwrite_config=None,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[gen_queries] total samples: {len(data)}")
    for sample in tqdm(data, desc="gen_queries"):
        sample_id = sample.get("id")
        video_rel = sample["video"]
        out_json = os.path.join(args.out_dir, f"{sample_id}_queries.json")

        if (not args.no_cache) and os.path.exists(out_json):
            continue

        video_path = os.path.join(args.video_root, video_rel)
        frames, shot_infos = sample_frames_for_query_stage(
            video_path, args.query_stage_frames, args.shot_based
        )
        video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(
            device=device, dtype=dtype
        )

        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], build_prompt(sample, shot_infos))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=[video_tensor],
                modalities=["video"],
                do_sample=False,
                temperature=0.0,
                max_new_tokens=512,
                use_cache=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        sequences = outputs.sequences
        raw_text = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip()
        token_stats: List[Dict[str, Any]] = []
        avg_logprob = None

        if outputs.scores:
            gen_token_count = len(outputs.scores)
            if gen_token_count > 0:
                gen_token_ids = sequences[0, -gen_token_count:]
            else:
                gen_token_ids = sequences.new_empty((0,), dtype=torch.long)
            logprob_sum = 0.0
            for step, (logits, token_id) in enumerate(zip(outputs.scores, gen_token_ids)):
                log_probs = F.log_softmax(logits[0], dim=-1)
                logprob = float(log_probs[token_id].item())
                logprob_sum += logprob
                token_text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
                token_stats.append(
                    {
                        "step": step,
                        "token_id": int(token_id),
                        "token": token_text,
                        "logprob": logprob,
                        "prob": float(math.exp(logprob)),
                    }
                )
            if token_stats:
                avg_logprob = logprob_sum / len(token_stats)
        parsed = safe_json_parse(raw_text)
        queries = parsed.get("queries") or []

        cleaned: List[Dict[str, Any]] = []
        for idx, item in enumerate(queries):
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
            else:
                text = str(item).strip()
            if not text:
                continue
            cleaned.append({"rank": len(cleaned) + 1, "text": text})
            if len(cleaned) >= args.num_queries:
                break

        if not cleaned:
            fallback = [f"Scene {i + 1}: {sample['question']}" for i in range(args.num_queries)]
            cleaned = [{"rank": i + 1, "text": text} for i, text in enumerate(fallback)]

        result = {
            "id": sample_id,
            "video": video_rel,
            "question": sample["question"],
            "queries": cleaned,
            "shot_based": args.shot_based,
            "raw": raw_text,
        }
        if shot_infos:
            result["shots"] = shot_infos
        if token_stats:
            result["token_stats"] = {
                "avg_logprob": avg_logprob,
                "tokens": token_stats,
            }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
