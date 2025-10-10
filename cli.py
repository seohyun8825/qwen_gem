import argparse
import json
import os


import numpy as np
import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from gem.config import DEFAULT_MAX_FRAMES, DEVICE
from gem.gem_core import gem_whitebox
from gem.heatmap import token_sim_heatmap
from gem.io_utils import bgr, load_dataset_json, read_video_frames
from gem.prompt import split_action_prompt
from gem.text_encoder import TextEncoder
from gem.vision_introspect import ViTWhitebox
from gem.vis import save_heatmaps_and_peaks, save_meta


def _locate_mm_projector(model) -> torch.nn.Module | None:
    """Try to locate the multimodal projector within a LLaVA-style model."""
    visited = set()
    candidates = []

    def enqueue(obj):
        if obj is None:
            return
        identifier = id(obj)
        if identifier in visited:
            return
        visited.add(identifier)
        candidates.append(obj)

    enqueue(model)
    for attr in ("model", "module", "backbone", "language_model"):
        enqueue(getattr(model, attr, None))

    get_model = getattr(model, "get_model", None)
    if callable(get_model):
        try:
            base = get_model()
            enqueue(base)
            for attr in ("model", "module", "backbone"):
                enqueue(getattr(base, attr, None))
        except Exception:
            pass

    for candidate in candidates:
        projector = getattr(candidate, "mm_projector", None)
        if projector is not None:
            return projector
    return None


def _apply_mm_projector(model, tokens: torch.Tensor) -> torch.Tensor:
    projector = _locate_mm_projector(model)
    if projector is None:
        return tokens
    param = next(projector.parameters(), None)
    if param is not None:
        if tokens.device != param.device or tokens.dtype != param.dtype:
            tokens = tokens.to(device=param.device, dtype=param.dtype)
    return projector(tokens)


def _preprocess_frames(image_processor, frames_rgb: np.ndarray) -> torch.Tensor:
    if frames_rgb.ndim != 4:
        raise ValueError(f"Expected frames shape [T, H, W, C], got {frames_rgb.shape}")

    pil_frames = []
    for idx, frame in enumerate(frames_rgb):
        if frame.ndim == 3 and frame.shape[-1] == 3:
            img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        elif frame.ndim == 3 and frame.shape[0] == 3:
            img = Image.fromarray(np.moveaxis(frame, 0, -1).astype(np.uint8)).convert("RGB")
        else:
            raise ValueError(f"Unsupported frame shape at index {idx}: {frame.shape}")
        pil_frames.append(img)

    batch = image_processor.preprocess(pil_frames, return_tensors="pt")
    return batch["pixel_values"]


def run_one_video(
    vid_path: str,
    question: str,
    model,
    image_processor,
    text_encoder: TextEncoder,
    out_dir: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> None:
    frames_rgb = read_video_frames(vid_path, max_frames=max_frames, force_uniform=True)
    frames_bgr = bgr(frames_rgb)
    pixel_values = _preprocess_frames(image_processor, frames_rgb).to(DEVICE)

    prompt_verb, prompt_object, prompt_action = split_action_prompt(question)
    if os.environ.get("GEM_DEBUG_TEXT", "").lower() in {"1", "true", "yes"}:
        print("[GEM][run] prompts:", prompt_verb, "|", prompt_object, "|", prompt_action)
    try:
        text_embeddings = text_encoder.encode([prompt_verb, prompt_object, prompt_action])
    except Exception as exc:
        print("[GEM][ERR] text_encoder.encode failed:", exc)
        print("[GEM][ERR] prompts:", [prompt_verb, prompt_object, prompt_action])
        raise
    emb_verb = text_embeddings[0]
    emb_object = text_embeddings[1]
    emb_action = text_embeddings[2]

    vit = ViTWhitebox(model, device=DEVICE)
    vit_outputs = vit.encode_frames(pixel_values)
    layers = vit_outputs["layers"]
    qkv = vit_outputs["qkv"]
    grid = vit_outputs["grid"]
    has_cls = vit_outputs["has_cls"]

    disable_whitebox = os.environ.get("GEM_DISABLE_WHITEBOX", "").lower() in {"1", "true", "yes"}
    if disable_whitebox:
        fused_tokens = layers[-1]
        layer_weights = None
    else:
        gem_outputs = gem_whitebox(layers, qkv, has_cls, text_eos=emb_action)
        fused_tokens = gem_outputs["O_comb"]
        layer_weights = gem_outputs["w_comb"]

    fused_tokens = _apply_mm_projector(model, fused_tokens).detach()
    target_device = fused_tokens.device
    target_dtype = fused_tokens.dtype
    emb_verb = emb_verb.to(device=target_device, dtype=target_dtype)
    emb_object = emb_object.to(device=target_device, dtype=target_dtype)
    emb_action = emb_action.to(device=target_device, dtype=target_dtype)

    heatmap_verb = token_sim_heatmap(fused_tokens, emb_verb, has_cls, grid).to(torch.float32).cpu().numpy()
    heatmap_object = token_sim_heatmap(fused_tokens, emb_object, has_cls, grid).to(torch.float32).cpu().numpy()
    heatmap_action = token_sim_heatmap(fused_tokens, emb_action, has_cls, grid).to(torch.float32).cpu().numpy()

    os.makedirs(out_dir, exist_ok=True)
    no_decompose = os.environ.get("GEM_NO_DECOMPOSE") == "1"
    if no_decompose:
        heatmaps = {"no_decomposed": heatmap_action}
        labels = {"no_decomposed": question}
        weights = None
    else:
        heatmaps = {"verb": heatmap_verb, "object": heatmap_object, "action": heatmap_action}
        labels = {"verb": prompt_verb, "object": prompt_object, "action": prompt_action}
        weights = None
    save_heatmaps_and_peaks(out_dir, frames_bgr, heatmaps, weights=weights, labels=labels)
    save_meta(
        out_dir,
        {
            "video_path": vid_path,
            "question": question,
            "prompts": {
                "verb": prompt_verb,
                "object": prompt_object,
                "action": prompt_action,
            },
            "grid": grid,
            "has_cls": has_cls,
            "w_comb": layer_weights,
            "whitebox_enabled": not disable_whitebox,
            "device": DEVICE,
        },
    )

    vit.remove_hooks()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_base", default=None)
    parser.add_argument("--model_name", default="llava_qwen")
    parser.add_argument("--data_path", required=False)
    parser.add_argument("--video_root", required=False)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--topk_frames", type=int, default=8)
    parser.add_argument("--one_video", default=None)
    parser.add_argument("--one_question", default=None)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=args.model_name,
        torch_dtype="bfloat16" if torch.cuda.is_available() else "float32",
        device_map="auto",
    )
    text_encoder = TextEncoder(model=model, tokenizer=tokenizer, device=DEVICE)

    if args.one_video:
        out_dir = os.path.join(args.results_dir, "debug_one")
        run_one_video(
            args.one_video,
            args.one_question or "flip egg",
            model,
            image_processor,
            text_encoder,
            out_dir,
            args.max_frames,
        )
        print(f"[OK] saved to {out_dir}")
        return

    if not args.data_path or not args.video_root:
        raise ValueError("dataset 모드에는 --data_path, --video_root 필요")

    data = load_dataset_json(args.data_path)

    for sample in data:
        vid_path = os.path.join(args.video_root, sample["video"])
        question = sample.get("question") or sample.get("caption") or "describe action"
        raw_id = sample.get("id")
        if raw_id is None:
            raw_id = os.path.splitext(os.path.basename(sample["video"]))[0]
        sample_id = str(raw_id)
        out_dir = os.path.join(args.results_dir, sample_id)

        if os.path.exists(os.path.join(out_dir, "meta.json")):
            continue

        try:
            run_one_video(
                vid_path,
                question,
                model,
                image_processor,
                text_encoder,
                out_dir,
                args.max_frames,
            )
            print("[OK]", sample_id)
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            import traceback

            print("[ERR]", sample_id, exc)
            traceback.print_exc()


if __name__ == "__main__":
    main()
