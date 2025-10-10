"""Vicuna-based evaluation pipeline using VTimeLLM checkpoints and CLIP features."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize
from tqdm import tqdm

# Disable FlashAttention kernels for compatibility
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
VTIMELLM_CANDIDATES = [
    PROJECT_ROOT / "VTimeLLM",
    PROJECT_ROOT.parent / "VTimeLLM",
    Path("/home/VTimeLLM"),
]
for candidate in VTIMELLM_CANDIDATES:
    if candidate.exists():
        sys.path.append(str(candidate))
        break
else:
    raise RuntimeError(
        "VTimeLLM repository not found. Expected one of: "
        + ", ".join(str(path) for path in VTIMELLM_CANDIDATES)
    )

from vtimellm.mm_utils import VideoExtractor  # type: ignore  # noqa: E402
from vtimellm.model.builder import load_pretrained_model  # type: ignore  # noqa: E402
from vtimellm.utils import disable_torch_init  # type: ignore  # noqa: E402

from inference_vicuna import load_clip_encoder as vicuna_load_clip_encoder  # type: ignore  # noqa: E402
from inference_vicuna import inference as vicuna_generate  # type: ignore  # noqa: E402

from llava.eval.infer import load_video, fuzzy_matching  # noqa: E402

warnings.filterwarnings("ignore")

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:  # pragma: no cover
    from PIL import Image

    BICUBIC = Image.BICUBIC

try:  # pragma: no cover
    from icecream import ic

    ic.configureOutput(prefix="[vicuna] ")
except ImportError:  # pragma: no cover
    def ic(*args, **kwargs):
        print(*args)


def build_prompt(
    sample: Dict[str, Any],
    dataset_name: str,
    *,
    frame_time: str | None = None,
    video_time: float | None = None,
    max_frames: int | None = None,
    include_time: bool = False,
    include_options: bool = True,
    override_prompt: str | None = None,
) -> str:
    if override_prompt is not None:
        return override_prompt.strip()

    prompt_lines: List[str] = []
    if include_time and frame_time is not None and video_time is not None and max_frames is not None:
        prompt_lines.append(
            f"The video lasts for {video_time:.2f} seconds, and {max_frames} frames are uniformly sampled from it. "
            f"These frames are located at {frame_time}."
        )

    if dataset_name.lower() == "videomme":
        options = sample.get("candidates", [])
        if include_options and options:
            prompt_lines.append(
                "Select the best answer to the following multiple-choice question based on the video. "
                "Explain the reason for the answer too"
            )
            prompt_lines.append(sample["question"])
            for option in options:
                prompt_lines.append(option)
            prompt_lines.append("The best answer is:")
        else:
            prompt_lines.append(sample["question"])
            if options and not include_options:
                prompt_lines.append("Please answer freely (choices hidden).")
    else:
        prompt_lines.append(sample.get("question", ""))

    return "\n".join(prompt_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", type=str, default="checkpoints/checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="checkpoints/checkpoints/lmsys-vicuna-7b-v1.5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="checkpoints/checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="checkpoints/checkpoints/vtimellm-vicuna-v1-5-7b-stage3")
    parser.add_argument("--dataset_name", type=str, default="VideoMME")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--max_frames_num", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--test_ratio", type=float, default=1.0)
    parser.add_argument("--use_time_ins", action="store_true")
    parser.add_argument("--cals_acc", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omit_options", action="store_true", help="Do not include multiple-choice options in the prompt")
    parser.add_argument("--prompt", type=str, default=None, help="Override the question/prompt with custom text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    disable_torch_init()

    ns = SimpleNamespace(
        model_base=args.model_base,
        clip_path=args.clip_path,
        pretrain_mm_mlp_adapter=args.pretrain_mm_mlp_adapter,
    )
    tokenizer, model, _ = load_pretrained_model(ns, args.stage2, args.stage3)
    model = model.cuda().to(torch.float16)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    clip_model = vicuna_load_clip_encoder(args.clip_path, device="cuda")
    preprocess = Compose(
        [
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    video_loader = VideoExtractor(N=args.max_frames_num)

    with open(args.data_path, "r") as f:
        dataset = json.load(f)

    if isinstance(dataset, dict):
        dataset = list(dataset.values())

    random.shuffle(dataset)
    num_samples = int(len(dataset) * args.test_ratio)
    dataset = dataset[:num_samples]

    results_dir = Path(args.results_dir)
    outputs_dir = results_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    device = torch.device("cuda")

    for sample in tqdm(dataset, desc="Evaluating"):
        video_path = os.path.join(args.video_root, sample["video"])
        _, frames = video_loader.extract({"id": None, "video": video_path})
        if frames is None or frames.numel() <= 1:
            warnings.warn(f"Video {video_path} could not be processed.")
            continue

        frames = frames.float().div_(255.0)
        frames = preprocess(frames)
        clip_dtype = next(clip_model.parameters()).dtype
        frames = frames.to(device=device, dtype=clip_dtype)
        with torch.no_grad():
            clip_features = clip_model.encode_image(frames)
        clip_features = clip_features.to(dtype=next(model.parameters()).dtype)

        frame_time = None
        video_time = None
        if args.use_time_ins and args.prompt is None:
            try:
                _, frame_time, video_time = load_video(
                    video_path,
                    args.max_frames_num,
                    fps=1,
                    force_sample=True,
                )
            except Exception:
                frame_time = None
                video_time = None

        prompt = build_prompt(
            sample,
            args.dataset_name,
            frame_time=frame_time,
            video_time=video_time,
            max_frames=args.max_frames_num,
            include_time=args.use_time_ins and args.prompt is None,
            include_options=not args.omit_options,
            override_prompt=args.prompt,
        )
        query = f"<video>\n{prompt}"
        answer = vicuna_generate(
            model,
            clip_features,
            query,
            tokenizer,
        )
        sample_out = dict(sample)
        sample_out["prediction"] = answer

        gt = sample_out.get("answer")
        score = None
        if gt is not None:
            score = 1 if fuzzy_matching(answer) == gt else 0
            if args.cals_acc:
                total += 1
                correct += score

        sample_id = sample.get("id") or Path(sample.get("video", "")).stem
        ic(
            {
                "sample": sample_id,
                "gt": gt,
                "prediction": answer,
                "score": score,
            }
        )

        output_path = outputs_dir / f"{sample_id}.json"
        with open(output_path, "w") as wf:
            json.dump(sample_out, wf, indent=2, ensure_ascii=False)

    if args.cals_acc and total:
        accuracy = correct / total
        summary_path = results_dir / "summary.json"
        with open(summary_path, "w") as sf:
            json.dump({"accuracy": accuracy, "correct": correct, "total": total}, sf, indent=2)
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
