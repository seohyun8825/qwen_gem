import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import functools
import itertools
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import warnings
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm
from PIL import Image
from rapidfuzz import process as fuzzy_process, fuzz as fuzzy_fuzz
from sentence_transformers import SentenceTransformer, util as st_util
import spacy
import open_clip

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- Shot segmentation (PySceneDetect) ----
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import cv2

try:
    from ERF.patch_infer import maybe_build_engine, run_erf_sample
except ImportError:  # pragma: no cover
    maybe_build_engine = None
    run_erf_sample = None

try:  # pragma: no cover
    from icecream import ic

    ic.configureOutput(prefix="[qwen] ")
except ImportError:  # pragma: no cover
    def ic(*args, **kwargs):
        print(*args)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import rank0_print

# Prefer flash / mem-efficient attention kernels when the GPU stack supports them; otherwise fall back.
if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
    flash_available = getattr(torch.backends.cuda, "flash_sdp_available", lambda: False)()
    mem_available = getattr(torch.backends.cuda, "mem_efficient_sdp_available", lambda: False)()
    try:
        if flash_available or mem_available:
            torch.backends.cuda.enable_flash_sdp(flash_available)
            torch.backends.cuda.enable_mem_efficient_sdp(mem_available)
            torch.backends.cuda.enable_math_sdp(True)
            print(
                "[SDP] flash_sdp_available:", flash_available,
                "mem_efficient_sdp_available:", mem_available,
                "-> using flash:", torch.backends.cuda.flash_sdp_enabled(),
                "mem:", torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math:", torch.backends.cuda.math_sdp_enabled(),
            )
        else:
            raise RuntimeError("Flash/mem-efficient SDP not available")
    except Exception:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[SDP] Falling back to math kernels (flash/mem-efficient unavailable).")


warnings.filterwarnings("ignore")


def detect_shots(video_path: str, threshold: float = 27.0, min_scene_len: int = 15):
    """Return list of (start_frame, end_frame) shots."""
    vm = VideoManager([video_path])
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    vm.start()
    sm.detect_scenes(frame_source=vm)
    scene_list = sm.get_scene_list()
    vm.release()
    out = []
    for start, end in scene_list:
        first, last = start.get_frames(), end.get_frames()
        if last > first:
            out.append((first, last))
    return out


def read_frame(video_path: str, frame_idx: int):
    """Return RGB frame (H, W, 3)."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frm = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frm[:, :, ::-1]


_SPACY_NLP = None
_TEXT_ENCODER = None
_TEXT_ENCODER_NAME = "all-MiniLM-L6-v2"
_CLIP_CACHE: Dict[str, Any] = {}


def get_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP


def get_text_encoder():
    global _TEXT_ENCODER
    if _TEXT_ENCODER is None:
        _TEXT_ENCODER = SentenceTransformer(_TEXT_ENCODER_NAME)
    return _TEXT_ENCODER


def get_clip_bundle(device: torch.device):
    key = str(device)
    if key in _CLIP_CACHE:
        return _CLIP_CACHE[key]
    created = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    if isinstance(created, tuple):
        if len(created) == 3:
            model, _preprocess_train, preprocess = created
        elif len(created) == 2:
            model, preprocess = created
        else:
            raise ValueError("Unexpected return signature from open_clip.create_model_and_transforms")
    else:
        raise ValueError("open_clip.create_model_and_transforms returned unexpected object")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bundle = {"model": model, "preprocess": preprocess, "tokenizer": tokenizer}
    _CLIP_CACHE[key] = bundle
    return bundle


def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()


def build_query_text(sample: Dict[str, Any]) -> Tuple[str, List[str]]:
    question = (sample.get("question") or "").strip()
    choices = sample.get("candidates") or []
    if isinstance(choices, dict):
        choices = list(choices.values())
    formatted_choices = []
    for idx, choice in enumerate(choices):
        if isinstance(choice, dict):
            choice_text = choice.get("text") or choice.get("answer") or ""
        else:
            choice_text = str(choice)
        choice_text = choice_text.strip()
        if not choice_text:
            continue
        formatted_choices.append(choice_text)
    if formatted_choices:
        joined_choices = "\nChoices:\n" + "\n".join(formatted_choices)
    else:
        joined_choices = ""
    query_text = question + ("\n" + joined_choices if joined_choices else "")
    return query_text.strip(), formatted_choices


def compute_frame_relevance(
    img_rgb: np.ndarray,
    query_text: str,
    clip_bundle: Dict[str, Any],
    device: torch.device,
) -> float:
    pil_img = Image.fromarray(img_rgb)
    image_tensor = clip_bundle["preprocess"](pil_img).unsqueeze(0).to(device)
    text_tokens = clip_bundle["tokenizer"]([query_text]).to(device)
    with torch.no_grad():
        image_feat = clip_bundle["model"].encode_image(image_tensor)
        text_feat = clip_bundle["model"].encode_text(text_tokens)
    image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    sim = (image_feat * text_feat).sum(dim=-1)
    return float(sim.cpu().item())


def extract_query_slots(query: str) -> List[str]:
    nlp = get_nlp()
    doc = nlp(query)
    slots: List[str] = []
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "ADJ", "NUM"}:
            slots.append(token.lemma_.lower())
    return slots


def normalize_choice_label(choice_text: str) -> Tuple[str, str]:
    cleaned = choice_text.strip()
    match = re.match(r"^\s*([A-Z])[\.\)]\s*(.*)$", cleaned)
    if match:
        return match.group(1), match.group(2) or cleaned
    return cleaned, cleaned


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract JSON substring.
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            snippet = text[first:last + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
    return None


class QAIGraphState:
    ALPHA = 0.5
    BETA = 0.2
    GAMMA = 0.3

    def __init__(
        self,
        text_encoder: SentenceTransformer,
        query_text: str,
        query_embedding: np.ndarray,
        query_slots: List[str],
        choices: List[str],
        decay_tau: float = 45.0,
    ):
        self.text_encoder = text_encoder
        self.query_text = query_text
        self.query_embedding = query_embedding
        self.query_slots = query_slots
        self.decay_tau = decay_tau
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.edge_lookup: Dict[Tuple[str, str, str], int] = {}
        self.choices = {
            choice: {"score": 0.0, "evidence": []} for choice in choices
        }
        self.choice_embeddings = {
            choice: self.text_encoder.encode(choice, normalize_embeddings=True)
            for choice in choices
        }
        self.emb_cache: Dict[str, np.ndarray] = {}

    def encode_text(self, text: str) -> np.ndarray:
        cached = self.emb_cache.get(text)
        if cached is not None:
            return cached
        emb = self.text_encoder.encode(text, normalize_embeddings=True)
        self.emb_cache[text] = emb
        return emb

    def lexical_overlap(self, text: str) -> float:
        return fuzzy_fuzz.partial_ratio(text.lower(), self.query_text.lower()) / 100.0

    def slot_overlap(self, text: str) -> float:
        if not self.query_slots:
            return 0.0
        text_lower = text.lower()
        hits = sum(1 for slot in self.query_slots if slot and slot in text_lower)
        return hits / len(self.query_slots)

    def compute_item_relevance(self, text: str) -> float:
        emb = self.encode_text(text)
        sim = float(st_util.cos_sim(
            torch.from_numpy(np.array([emb])),
            torch.from_numpy(np.array([self.query_embedding]))
        )[0][0])
        lex = self.lexical_overlap(text)
        slot = self.slot_overlap(text)
        return max(
            0.0,
            min(1.0, self.ALPHA * sim + self.BETA * lex + self.GAMMA * slot),
        )

    def _canonicalize_name(self, name: str) -> str:
        if name in self.nodes:
            return name
        candidates = list(self.nodes.keys())
        if not candidates:
            return name
        best_match, score, _ = fuzzy_process.extractOne(
            name, candidates, scorer=fuzzy_fuzz.WRatio
        )
        if score >= 92:
            return best_match
        # Fallback to embedding sim
        name_emb = self.encode_text(name)
        cand_embs = [self.encode_text(c) for c in candidates]
        sims = st_util.cos_sim(
            torch.from_numpy(np.array([name_emb])),
            torch.from_numpy(np.array(cand_embs)),
        ).cpu().numpy()[0]
        idx = int(np.argmax(sims))
        if sims[idx] >= 0.78:
            return candidates[idx]
        return name

    def _node_struct(self, canonical: str) -> Dict[str, Any]:
        return self.nodes.setdefault(
            canonical,
            {
                "id": canonical,
                "attrs": defaultdict(dict),
                "aliases": set([canonical]),
                "relevance": 0.0,
                "support": 0,
                "timestamps": [],
                "sources": [],
            },
        )

    def update_node(
        self,
        original_name: str,
        attrs: Dict[str, Any],
        relevance: float,
        timestamp: int,
        frame_id: int,
        confidence: float,
    ) -> str:
        canonical = self._canonicalize_name(original_name)
        node = self._node_struct(canonical)
        node["aliases"].add(original_name)
        node["support"] += 1
        node["relevance"] = max(node["relevance"], relevance)
        node["timestamps"].append(timestamp)
        node["sources"].append(frame_id)
        weight = relevance * confidence
        for attr, value in (attrs or {}).items():
            if value in (None, "", []):
                continue
            value_str = str(value)
            attr_bucket = node["attrs"].setdefault(attr, {})
            attr_bucket[value_str] = attr_bucket.get(value_str, 0.0) + weight
        return canonical

    def update_edge(
        self,
        source: str,
        relation: str,
        target: str,
        relevance: float,
        timestamp: int,
        frame_id: int,
        confidence: float,
    ) -> Dict[str, Any]:
        key = (source, relation, target)
        weight = relevance * confidence
        if key in self.edge_lookup:
            edge = self.edges[self.edge_lookup[key]]
            edge["support"] += 1
            edge["relevance"] = max(edge["relevance"], relevance)
            edge["timestamps"].append(timestamp)
            edge["sources"].append(frame_id)
            edge["weight"] += weight
            edge["versions"].append(
                {"value": relation, "conf": confidence, "at": timestamp}
            )
            return edge
        # Check conflicting relations between same nodes
        alt_keys = [
            (source, e["rel"], target) for e in self.edges
            if e["source"] == source and e["target"] == target and e["rel"] != relation
        ]
        for alt_key in alt_keys:
            alt_idx = self.edge_lookup.get(alt_key)
            if alt_idx is not None:
                self.edges[alt_idx]["versions"].append(
                    {"value": relation, "conf": confidence, "at": timestamp}
                )
        edge = {
            "source": source,
            "rel": relation,
            "target": target,
            "relevance": relevance,
            "support": 1,
            "timestamps": [timestamp],
            "sources": [frame_id],
            "weight": weight,
            "versions": [{"value": relation, "conf": confidence, "at": timestamp}],
        }
        self.edge_lookup[key] = len(self.edges)
        self.edges.append(edge)
        return edge

    def update_choices(
        self,
        items: List[Dict[str, Any]],
        timestamp: int,
    ):
        if not self.choices:
            return
        for choice_text, choice_data in self.choices.items():
            emb = self.choice_embeddings[choice_text]
            score_gain = 0.0
            frame_evidence = []
            for item in items:
                item_emb = self.encode_text(item["text"])
                sim = float(st_util.cos_sim(
                    torch.from_numpy(np.array([item_emb])),
                    torch.from_numpy(np.array([emb])),
                )[0][0])
                weight = item["weight"]
                contribution = sim * weight
                if contribution > 0:
                    score_gain += contribution
                    frame_evidence.append({
                        "type": item["type"],
                        "id": item["id"],
                        "at": timestamp,
                        "contribution": contribution,
                    })
            choice_data["score"] += score_gain
            if frame_evidence:
                choice_data["evidence"].append({
                    "timestamp": timestamp,
                    "items": frame_evidence,
                })

    def finalize(self) -> Dict[str, Any]:
        finalized_nodes = []
        for name, node in self.nodes.items():
            attrs = {}
            for attr_name, values in node["attrs"].items():
                sorted_values = sorted(
                    values.items(), key=lambda kv: kv[1], reverse=True
                )
                attrs[attr_name] = [
                    {"value": val, "score": score} for val, score in sorted_values
                ]
            finalized_nodes.append(
                {
                    "id": name,
                    "attrs": attrs,
                    "aliases": sorted(node["aliases"]),
                    "relevance": node["relevance"],
                    "support": node["support"],
                    "timestamps": node["timestamps"],
                    "sources": node["sources"],
                }
            )

        finalized_edges = []
        for edge in self.edges:
            finalized_edges.append(
                {
                    "source": edge["source"],
                    "target": edge["target"],
                    "rel": edge["rel"],
                    "relevance": edge["relevance"],
                    "support": edge["support"],
                    "timestamps": edge["timestamps"],
                    "sources": edge["sources"],
                    "versions": edge["versions"],
                    "weight": edge["weight"],
                }
            )

        choices = {
            choice: {
                "score": data["score"],
                "evidence": data["evidence"],
            }
            for choice, data in self.choices.items()
        }

        return {
            "nodes": finalized_nodes,
            "edges": finalized_edges,
            "choices": choices,
        }

    def best_choice(self) -> Optional[str]:
        if not self.choices:
            return None
        ranked = sorted(
            self.choices.items(), key=lambda kv: kv[1]["score"], reverse=True
        )
        if not ranked:
            return None
        top_choice, top_data = ranked[0]
        if len(ranked) == 1:
            return top_choice
        second_score = ranked[1][1]["score"]
        if (top_data["score"] - second_score) < 0.05:
            return None
        return top_choice


def node_text_repr(node: Dict[str, Any]) -> str:
    name = node.get("name") or ""
    attrs = node.get("attrs") or {}
    attr_parts = []
    if isinstance(attrs, dict):
        for key, value in attrs.items():
            if value in (None, "", []):
                continue
            attr_parts.append(f"{key}:{value}")
    return (name + " " + " ".join(attr_parts)).strip()


def edge_text_repr(edge: Dict[str, Any]) -> str:
    src = edge.get("source") or edge.get("subject") or ""
    rel = edge.get("rel") or edge.get("relation") or ""
    tgt = edge.get("target") or edge.get("object") or ""
    return f"{src} {rel} {tgt}".strip()


def prepare_structured_prompt(query_text: str) -> str:
    instruction = (
        "You will see ONE image frame.\n"
        "Focus ONLY on facts relevant to the query; ignore unrelated details.\n"
        f"Query: \"{query_text}\"\n\n"
        "Return a JSON with this exact schema:\n"
        "{\n"
        '  "nodes": [{"name": str, "attrs": {"color": str?, "count": int?}}],\n'
        '  "edges": [{"source": str, "rel": str, "target": str}],\n'
        '  "confidence": float,\n'
        '  "note": str\n'
        "}\n"
        'If no relevant info, return {"nodes": [], "edges": [], "confidence": 0.0, "note": "no relevant info"}.\n'
        "Do not add any extra fields or text."
    )
    return instruction


def run_qaig_on_sample(
    sample: Dict[str, Any],
    args,
    tokenizer,
    model,
    image_processor,
    model_device,
    model_dtype,
) -> Dict[str, Any]:
    qaig_result = copy.deepcopy(sample)
    video_path = os.path.join(args.video_root, sample["video"])
    shots = detect_shots(
        video_path,
        threshold=args.shot_threshold,
        min_scene_len=args.shot_min_len,
    )
    if not shots:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        del vr
        shots = [(0, total_frames)]

    qaig_threshold = getattr(args, "qaig_frame_threshold", 0.25)
    qaig_conf_threshold = getattr(args, "qaig_conf_threshold", 0.5)
    decay_tau = getattr(args, "qaig_decay_tau", 45.0)

    query_text, choices = build_query_text(sample)
    text_encoder = get_text_encoder()
    query_embedding = text_encoder.encode(query_text, normalize_embeddings=True)
    query_slots = extract_query_slots(query_text)
    choice_entries = []
    for ch in choices:
        label, plain = normalize_choice_label(ch)
        choice_entries.append(
            {
                "label": label,
                "text": plain,
                "raw": ch,
            }
        )
    choice_texts = [c["text"] for c in choice_entries]

    graph_state = QAIGraphState(
        text_encoder=text_encoder,
        query_text=query_text,
        query_embedding=query_embedding,
        query_slots=query_slots,
        choices=choice_texts,
        decay_tau=decay_tau,
    )

    clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_bundle = get_clip_bundle(clip_device)
    shot_outputs: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []

    structured_prompt = prepare_structured_prompt(query_text)

    for shot_idx, (fs, fe) in enumerate(shots):
        rep = (fs + fe) // 2
        img_rgb = read_frame(video_path, rep)
        frame_relevance = compute_frame_relevance(
            img_rgb, query_text, clip_bundle, clip_device
        )
        if frame_relevance < qaig_threshold:
            logs.append(
                {
                    "shot_id": shot_idx,
                    "frame": rep,
                    "range_frames": [fs, fe],
                    "frame_relevance": frame_relevance,
                    "status": "skipped",
                    "reason": "low relevance",
                }
            )
            shot_outputs.append(
                {
                    "shot_id": shot_idx,
                    "frame": rep,
                    "range_frames": [fs, fe],
                    "frame_relevance": frame_relevance,
                    "note": "no relevant info",
                    "nodes": [],
                    "edges": [],
                    "confidence": 0.0,
                }
            )
            continue

        one = np.expand_dims(img_rgb, axis=0)
        video_tensor = image_processor.preprocess(one, return_tensors="pt")["pixel_values"]
        video_tensor = video_tensor.to(device=model_device, dtype=model_dtype)

        prompt_question = DEFAULT_IMAGE_TOKEN + structured_prompt
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], prompt_question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model_device)

        with torch.inference_mode():
            cont = model.generate(
                input_ids,
                images=[video_tensor],
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=min(args.max_new_tokens, 256),
                use_cache=False,
            )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

        del input_ids, cont, video_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        parsed = safe_json_parse(text_outputs)
        if not parsed:
            logs.append(
                {
                    "shot_id": shot_idx,
                    "frame": rep,
                    "range_frames": [fs, fe],
                    "frame_relevance": frame_relevance,
                    "status": "skipped",
                    "reason": "json_parse_failed",
                    "raw_output": text_outputs,
                }
            )
            shot_outputs.append(
                {
                    "shot_id": shot_idx,
                    "frame": rep,
                    "range_frames": [fs, fe],
                    "frame_relevance": frame_relevance,
                    "note": "parse error",
                    "nodes": [],
                    "edges": [],
                    "confidence": 0.0,
                }
            )
            continue

        nodes = parsed.get("nodes") or []
        edges = parsed.get("edges") or []
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        note = parsed.get("note") or ""

        if confidence < qaig_conf_threshold or (
            not nodes and not edges
        ) or note.lower().strip() == "no relevant info":
            logs.append(
                {
                    "shot_id": shot_idx,
                    "frame": rep,
                    "range_frames": [fs, fe],
                    "frame_relevance": frame_relevance,
                    "status": "skipped",
                    "reason": "low confidence" if confidence < qaig_conf_threshold else "no relevant info",
                    "confidence": confidence,
                    "raw_output": parsed,
                }
            )
            shot_outputs.append(
                {
                    "shot_id": shot_idx,
                    "frame": rep,
                    "range_frames": [fs, fe],
                    "frame_relevance": frame_relevance,
                    "note": "no relevant info",
                    "nodes": [],
                    "edges": [],
                    "confidence": confidence,
                }
            )
            continue

        node_map: Dict[str, str] = {}
        frame_items: List[Dict[str, Any]] = []

        for node in nodes:
            name = (node.get("name") or "").strip()
            if not name:
                continue
            attrs = node.get("attrs") or {}
            node_text = node_text_repr(node)
            relevance = graph_state.compute_item_relevance(node_text)
            node["relevance"] = relevance
            canonical = graph_state.update_node(
                original_name=name,
                attrs=attrs,
                relevance=relevance,
                timestamp=shot_idx,
                frame_id=rep,
                confidence=confidence,
            )
            node_map[name] = canonical
            frame_items.append(
                {
                    "type": "node",
                    "id": canonical,
                    "text": node_text,
                    "weight": relevance * confidence,
                }
            )

        for edge in edges:
            src = edge.get("source") or edge.get("subject") or ""
            tgt = edge.get("target") or edge.get("object") or ""
            rel = edge.get("rel") or edge.get("relation") or ""
            if not (src and tgt and rel):
                continue
            if src not in node_map:
                node_map[src] = graph_state.update_node(
                    original_name=src,
                    attrs={},
                    relevance=0.2,
                    timestamp=shot_idx,
                    frame_id=rep,
                    confidence=confidence,
                )
            if tgt not in node_map:
                node_map[tgt] = graph_state.update_node(
                    original_name=tgt,
                    attrs={},
                    relevance=0.2,
                    timestamp=shot_idx,
                    frame_id=rep,
                    confidence=confidence,
                )
            canonical_src = node_map[src]
            canonical_tgt = node_map[tgt]
            edge_text = edge_text_repr(edge)
            relevance = graph_state.compute_item_relevance(edge_text)
            edge["relevance"] = relevance
            updated_edge = graph_state.update_edge(
                source=canonical_src,
                relation=rel,
                target=canonical_tgt,
                relevance=relevance,
                timestamp=shot_idx,
                frame_id=rep,
                confidence=confidence,
            )
            frame_items.append(
                {
                    "type": "edge",
                    "id": f"{canonical_src}->{rel}->{canonical_tgt}",
                    "text": edge_text,
                    "weight": relevance * confidence,
                }
            )

        graph_state.update_choices(frame_items, timestamp=shot_idx)

        shot_outputs.append(
            {
                "shot_id": shot_idx,
                "frame": rep,
                "range_frames": [fs, fe],
                "frame_relevance": frame_relevance,
                "note": note,
                "nodes": nodes,
                "edges": edges,
                "confidence": confidence,
            }
        )
        logs.append(
            {
                "shot_id": shot_idx,
                "frame": rep,
                "range_frames": [fs, fe],
                "frame_relevance": frame_relevance,
                "status": "updated",
                "confidence": confidence,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
            }
        )

    final_graph = graph_state.finalize()
    best_choice_text = graph_state.best_choice()

    choice_outputs = []
    for c in choice_entries:
        data = final_graph["choices"].get(c["text"], {"score": 0.0, "evidence": []})
        choice_outputs.append(
            {
                "label": c["label"],
                "text": c["text"],
                "raw": c["raw"],
                "score": data.get("score", 0.0),
                "evidence": data.get("evidence", []),
            }
        )

    qaig_result.update(
        {
            "query": query_text,
            "graph": final_graph,
            "choices": choice_outputs,
            "logs": logs,
            "shot_outputs": shot_outputs,
            "qaig_config": {
                "frame_threshold": qaig_threshold,
                "confidence_threshold": qaig_conf_threshold,
                "decay_tau": decay_tau,
            },
        }
    )
    if best_choice_text:
        match_entry = next((c for c in choice_outputs if c["text"] == best_choice_text), None)
        if match_entry:
            qaig_result["prediction"] = match_entry["label"]
            qaig_result["prediction_text"] = match_entry["text"]
        else:
            qaig_result["prediction"] = best_choice_text
    return qaig_result


def load_frames_by_indices(video_path, indices):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 30.0
    if not indices:
        midpoint = total_frame_num // 2 if total_frame_num else 0
        indices = [midpoint]
    max_idx = max(total_frame_num - 1, 0)
    safe_indices = [min(max(0, int(idx)), max_idx) for idx in indices]
    frames = vr.get_batch(safe_indices).asnumpy()
    frame_times = [idx / max(1e-6, avg_fps) for idx in safe_indices]
    frame_time = ",".join(f"{t:.2f}s" for t in frame_times)
    video_time = total_frame_num / max(1e-6, avg_fps)
    return frames, frame_time, video_time


def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def get_options_letter(len_options):
    if len_options==2:
        return '(A or B)'
    elif len_options==3:
        return '(A, B or C)'
    elif len_options==4:
        return '(A, B, C or D)'
    elif len_options==5:
        return '(A, B, C, D, or E)'
    else:
        raise NotImplementedError

def get_prompt(dataset_name, sample, conv_template="qwen_1_5", video_time=None, num_frames=None, frame_time=None):
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    if video_time:
        prompt = f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly sampled from it. These frames are located at {frame_time}.\n"
    else:
        prompt = ""

    if dataset_name in ['VSI']:
        prompt += "These are frames of a video.\n"
        prompt += sample["question"] + "\n"
        if 'candidates' in sample:
            for op in sample["candidates"]:
                prompt += f"{op}\n"
            prompt += "Answer with the option's letter from the given choices directly."
        else:
            prompt += "Please answer the question using a single word or phrase."
    elif dataset_name in ['MovieChat']:
        if video_time is None:
            prompt += "These are frames of a video.\n"
        if 'time' in sample:
            timestamp = round(sample['time']/sample['fps'], 2)
            prompt += f"At time {timestamp}s, "
        prompt += sample["question"] + "\n"
        prompt += "Please answer the question using a single word, phrase, or sentence."
        #prompt += "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
    else:
        options_letter = get_options_letter(len(sample['candidates']))
        prompt += f"Select the best answer to the following multiple-choice question based on the video. Respond with only the letter {options_letter} of the correct option.\n"
        prompt += sample["question"] + "\n"
        for op in sample["candidates"]:
            prompt += f"{op}\n"
        prompt += f"The best answer is:"
        
    question = DEFAULT_IMAGE_TOKEN + prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def run(rank, world_size, args):
    torch.cuda.set_device(rank)

    rank0_print("Loadind dataset from", args.data_path)
    with open(args.data_path, "r") as f:
        dataset = json.load(f)
     
    random.shuffle(dataset)

    num_samples = int(len(dataset) * args.test_ratio)
    dataset = dataset[rank:num_samples:world_size]
    rank0_print(f"Total samples: {num_samples}")
    print(f"Samples in rank {rank}: {len(dataset)}")

    device_map = "auto"
    if args.multiprocess or world_size > 1:
        device_map = {"": torch.device(f"cuda:{rank}")}

    overwrite_cfg = None
    if isinstance(args.temporal_pooling, int) and args.temporal_pooling and args.temporal_pooling > 1:
        overwrite_cfg = {"temporal_pooling": args.temporal_pooling}

    tokenizer, model, image_processor, max_length = load_pretrained_model(
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

    if maybe_build_engine is None:
        if args.erf_enable:
            raise ImportError("ERF module not available but --erf_enable is set")
        erf_engine = None
    else:
        erf_engine = maybe_build_engine(args, tokenizer, model, image_processor)


    result_list = []
    for cnt, sample in enumerate(tqdm(dataset)):
        if args.shot_based:
            outputs_dir = os.path.join(args.results_dir, "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            if getattr(args, "qaig", False):
                qaig_path = os.path.join(outputs_dir, f"{sample['id']}_qaig.json")
                if os.path.exists(qaig_path) and not args.no_cache:
                    with open(qaig_path, "r") as f:
                        cached_sample = json.load(f)
                    result_list.append(cached_sample)
                    continue

                qaig_result = run_qaig_on_sample(
                    sample=sample,
                    args=args,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    model_device=model_device,
                    model_dtype=model_dtype,
                )

                qaig_result["shot_outputs_path"] = []
                for shot in qaig_result.get("shot_outputs", []):
                    shot_path = os.path.join(
                        outputs_dir, f"{sample['id']}_shot{shot['shot_id']}.json"
                    )
                    with open(shot_path, "w") as sf:
                        json.dump(shot, sf, indent=2)
                    qaig_result["shot_outputs_path"].append(os.path.relpath(shot_path, args.results_dir))

                with open(qaig_path, "w") as f:
                    json.dump(qaig_result, f, indent=2)

                result_list.append(qaig_result)
                continue
            else:
                video_path = os.path.join(args.video_root, sample["video"])
                shot_files = []
                if not args.no_cache:
                    if os.path.isdir(outputs_dir):
                        shot_files = sorted(
                            [
                                fp for fp in os.listdir(outputs_dir)
                                if fp.startswith(f"{sample['id']}_shot") and fp.endswith(".json")
                            ]
                        )
                if shot_files and not args.no_cache:
                    cached = []
                    for fname in shot_files:
                        path = os.path.join(outputs_dir, fname)
                        with open(path, "r") as f:
                            cached.append(json.load(f))
                    result_list.extend(cached)
                    continue

                shots = detect_shots(
                    video_path,
                    threshold=args.shot_threshold,
                    min_scene_len=args.shot_min_len,
                )
                if not shots:
                    vr = VideoReader(video_path, ctx=cpu(0))
                    total_frames = len(vr)
                    del vr
                    shots = [(0, total_frames)]

                shot_outputs = []
                for sid, (fs, fe) in enumerate(shots):
                    rep = (fs + fe) // 2
                    img_rgb = read_frame(video_path, rep)
                    one = np.expand_dims(img_rgb, axis=0)
                    video_tensor = image_processor.preprocess(one, return_tensors="pt")["pixel_values"]
                    video_tensor = video_tensor.to(device=model_device, dtype=model_dtype)

                    prompt_question = DEFAULT_IMAGE_TOKEN + args.caption_prompt
                    conv = copy.deepcopy(conv_templates["qwen_1_5"])
                    conv.append_message(conv.roles[0], prompt_question)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    input_ids = tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    ).unsqueeze(0).to(model_device)

                    with torch.inference_mode():
                        cont = model.generate(
                            input_ids,
                            images=[video_tensor],
                            modalities=["video"],
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=min(args.max_new_tokens, 128),
                            use_cache=False,
                        )
                    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

                    shot_json = {
                        "id": f"{sample.get('id')}_shot{sid}",
                        "video": sample["video"],
                        "shot_id": sid,
                        "frame": rep,
                        "range_frames": [fs, fe],
                        "prediction": text_outputs,
                        "question": "CAPTION",
                    }
                    shot_outputs.append(shot_json)

                    per_shot_save = os.path.join(outputs_dir, f"{shot_json['id']}.json")
                    with open(per_shot_save, "w") as f:
                        json.dump(shot_json, f, indent=4)

                    del input_ids, cont, video_tensor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                result_list.extend(shot_outputs)
                continue

        sample_save_path = f"{args.results_dir}/outputs/{sample['id']}.json"
        load_cached = os.path.exists(sample_save_path) and not args.erf_enable and not args.no_cache
        if load_cached:
            with open(sample_save_path, "r") as f:
                sample = json.load(f)
        else:
            video_path = os.path.join(args.video_root, sample["video"])
            use_selected = False
            selected_indices = []
            if args.selected_frames_root:
                sample_id = sample.get("id")
                if sample_id is not None:
                    selected_path = os.path.join(
                        args.selected_frames_root, f"{sample_id}_selected.json"
                    )
                    if os.path.exists(selected_path):
                        with open(selected_path, "r", encoding="utf-8") as sf:
                            sel_payload = json.load(sf)
                        selected_entries = sel_payload.get("selected") or []
                        selected_indices = [
                            int(entry.get("frame_index", 0)) for entry in selected_entries
                        ]
                        if args.selected_frames_topn and args.selected_frames_topn > 0:
                            selected_indices = selected_indices[: args.selected_frames_topn]
                        if selected_indices:
                            use_selected = True

            if use_selected:
                video_np, frame_time, video_time = load_frames_by_indices(video_path, selected_indices)
            else:
                video_np, frame_time, video_time = load_video(
                    video_path, args.max_frames_num, fps=1, force_sample=True
                )
            video_tensor = image_processor.preprocess(video_np, return_tensors="pt")[
                "pixel_values"
            ]
            video_tensor = video_tensor.to(device=model_device, dtype=model_dtype)
            video_list = [video_tensor]

            if erf_engine is not None:
                erf_artifacts = run_erf_sample(
                    erf_engine,
                    sample=sample,
                    dataset_name=args.dataset_name,
                    base_prompt_builder=get_prompt,
                    max_frames_num=args.max_frames_num,
                    video_time=video_time,
                    frame_time=frame_time,
                    video_tensor=video_tensor,
                    results_dir=args.results_dir,
                )
                if erf_artifacts and erf_artifacts.get("rounds"):
                    final_round = erf_artifacts["rounds"][-1]
                    weights = final_round.get("weights", [])
                    best_idx = erf_artifacts.get("best_index", 0)
                    cand_logs = final_round.get("cand_logs") or []
                    best_log = cand_logs[best_idx] if 0 <= best_idx < len(cand_logs) else None
                    cons_dbg = best_log["score"]["cons"] if best_log else 0.0
                    evid_dbg = best_log.get("evid_sim", 0.0) if best_log else 0.0
                    cal_dbg = best_log["score"].get("cal", 0.0) if best_log else 0.0
                    weights_str = ",".join(f"{w:.2f}" for w in weights)
                    print(
                        f"[ERF] id={sample.get('id')} best={erf_artifacts.get('final_prediction')} "
                        f"w=[{weights_str}] cons={cons_dbg:.2f} "
                        f"evid={evid_dbg:.2f} cal={cal_dbg:.2f}"
                    )
            else:
                if args.use_time_ins:
                    prompt_question = get_prompt(
                        args.dataset_name,
                        sample,
                        video_time=video_time,
                        num_frames=args.max_frames_num,
                        frame_time=frame_time,
                    )
                else:
                    prompt_question = get_prompt(args.dataset_name, sample)

                input_ids = tokenizer_image_token(
                    prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(model_device)

                try:
                    with torch.inference_mode():
                        cont = model.generate(
                            input_ids,
                            images=video_list,
                            modalities=["video"],
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=False,
                        )
                except RuntimeError as exc:
                    if "no kernel found" in str(exc).lower() or "cutlass" in str(exc).lower():
                        torch.backends.cuda.enable_flash_sdp(False)
                        torch.backends.cuda.enable_mem_efficient_sdp(False)
                        torch.backends.cuda.enable_math_sdp(True)
                        torch.cuda.empty_cache()
                        with torch.inference_mode():
                            cont = model.generate(
                                input_ids,
                                images=video_list,
                                modalities=["video"],
                                do_sample=False,
                                temperature=0,
                                max_new_tokens=args.max_new_tokens,
                                use_cache=False,
                            )
                    else:
                        raise
                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                sample["prediction"] = text_outputs

                del input_ids, cont

            del video_tensor, video_list, video_np
            torch.cuda.empty_cache()

            with open(sample_save_path, "w") as f:
                json.dump(sample, f, indent=4)

        result_list.append(sample)
        gt = sample.get("answer")
        score = None
        if gt is not None:
            score = 1 if fuzzy_matching(sample["prediction"]) == gt else 0
        ic({
            "idx": cnt,
            "id": sample.get("id"),
            "gt": gt,
            "prediction": sample.get("prediction"),
            "score": score,
        })
        if gt is None:
            print(cnt, "Pred:", sample["prediction"])
    
    return result_list


def main():
    parser = argparse.ArgumentParser(description="Run Inference")

    # Model
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--max_frames_num", type=int, default=16)
    # Use values >1 to pool across the temporal dimension; 0 keeps the model default.
    parser.add_argument("--temporal_pooling", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--conv_template", type=str, default="qwen_1_5")
    parser.add_argument("--use_time_ins", action="store_true")
    parser.add_argument("--lora_alpha", type=int, default=None)

    # Data
    parser.add_argument("--dataset_name", type=str, default="VideoMME")
    parser.add_argument("--data_path", type=str, default="/mnt/bum/mmiemon/datasets/Video-MME/formatted_dataset.json")
    parser.add_argument("--video_root", type=str, default="/mnt/bum/mmiemon/datasets/Video-MME/videos/data")
    parser.add_argument("--results_dir", type=str, default="/mnt/bum/mmiemon/LLaVA-NeXT/results/llava_video/VideoMME")
    parser.add_argument("--selected_frames_root", type=str, default=None,
                        help="Optional directory containing <sample_id>_selected.json files to drive frame selection.")
    parser.add_argument("--selected_frames_topn", type=int, default=0,
                        help="If >0, limit number of selected frames loaded for inference (0 means all).")
    parser.add_argument("--test_ratio", type=float, default=1)
    parser.add_argument("--multiprocess", action="store_true")
    parser.add_argument("--cals_acc", action="store_true")
    parser.add_argument("--erf_enable", action="store_true")
    parser.add_argument("--erf_K", type=int, default=6)
    parser.add_argument("--erf_rounds", type=int, default=2)
    parser.add_argument("--erf_tau", type=float, default=0.8)
    parser.add_argument("--erf_weights", type=str, default="0.4,0.4,0.2,0.0")
    parser.add_argument("--erf_debug", action="store_true")
    parser.add_argument("--no_cache", action="store_true", help="Ignore existing cached outputs")
    parser.add_argument("--shot_based", action="store_true",
                        help="Use PySceneDetect to split shots and caption exactly 1 frame per shot.")
    parser.add_argument("--shot_threshold", type=float, default=27.0,
                        help="PySceneDetect ContentDetector threshold (higher = fewer shots).")
    parser.add_argument("--shot_min_len", type=int, default=15,
                        help="Minimum scene length in frames for PySceneDetect.")
    parser.add_argument(
        "--caption_prompt",
        type=str,
        default="Describe this image in one concise sentence.",
        help="Prompt for captioning a single representative frame.",
    )
    parser.add_argument("--qaig", action="store_true",
                        help="Enable Query-Adaptive Incremental Graphing pipeline for shot-based runs.")
    parser.add_argument("--qaig_frame_threshold", type=float, default=0.25,
                        help="Frame-level relevance threshold (CLIP cosine).")
    parser.add_argument("--qaig_conf_threshold", type=float, default=0.5,
                        help="Minimum confidence from structured extractor to accept updates.")
    parser.add_argument("--qaig_decay_tau", type=float, default=45.0,
                        help="Temporal decay tau used when updating graph statistics.")

    args = parser.parse_args()
    if args.model_base == "None":
        args.model_base = None

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/outputs", exist_ok=True)


    if args.multiprocess:
        mp.set_start_method("spawn")
        print(f"started benchmarking")
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        print("World size", world_size)
        with Pool(world_size) as pool:
            func = functools.partial(run, args=args, world_size=world_size)
            result_lists = pool.map(func, range(world_size))

        print("finished running")
        result_list = [res for res in itertools.chain(*result_lists)]
    else:
        result_list = run(0, world_size=1, args=args)
    

    if args.cals_acc:
        results = {"all": {"correct": 0, "total": 0}}
        for sample in result_list:
            if "answer" not in sample:
                continue
            results["all"]["total"] += 1
            if "question_type" in sample:
                if sample["question_type"] not in results:
                    results[sample["question_type"]] = {"correct": 0, "total": 0}
                results[sample["question_type"]]["total"] += 1
                
            if sample["answer"].lower()==fuzzy_matching(sample["prediction"]).lower():
                results["all"]["correct"] += 1
                if "question_type" in sample:
                    results[sample["question_type"]]["correct"] += 1

        for key in results:
            results[key]["accuracy"] = results[key]["correct"] / results[key]["total"]

        print(results)

        with open(os.path.join(args.results_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
