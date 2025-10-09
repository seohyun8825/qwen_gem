import os
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import BatchEncoding

from .config import DEVICE

DEBUG_TEXT = os.environ.get("GEM_DEBUG_TEXT", "").lower() in {"1", "true", "yes"}


def _pad_batch(sequences: Iterable[Sequence[int] | torch.Tensor], pad_token_id: int, device: str) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    max_len = 0
    for idx, seq in enumerate(sequences):
        if torch.is_tensor(seq):
            tensor = seq.long().flatten()
        else:
            tensor = torch.tensor(list(seq), dtype=torch.long)
        tensors.append(tensor)
        max_len = max(max_len, tensor.numel())
        if DEBUG_TEXT:
            print(f"[GEM][text] seq[{idx}] len={tensor.numel()}")
    if not tensors:
        raise ValueError("Empty token sequence batch")
    padded = []
    for tensor in tensors:
        if tensor.numel() < max_len:
            tensor = F.pad(tensor, (0, max_len - tensor.numel()), value=pad_token_id)
        padded.append(tensor)
    batch = torch.stack(padded, dim=0).to(device)
    if DEBUG_TEXT:
        print("[GEM][text] padded batch shape:", batch.shape)
    return batch


class TextEncoder:
    def __init__(self, model, tokenizer, device: str = DEVICE):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        get_model = getattr(model, "get_model", None)
        base_model = None
        if callable(get_model):
            try:
                base_model = get_model()
            except Exception as exc:
                if DEBUG_TEXT:
                    print("[GEM][text] get_model() failed:", exc)
        if base_model is None and hasattr(model, "model"):
            base_model = getattr(model, "model")
        if base_model is None:
            base_model = model

        candidates = []
        for source, attr in (
            (model, "embed_tokens"),
            (base_model, "embed_tokens"),
            (getattr(base_model, "model", None), "embed_tokens"),
            (getattr(model, "language_model", None), "embed_tokens"),
        ):
            if source is None:
                continue
            emb = getattr(source, attr, None)
            if emb is not None:
                candidates.append(emb)
        if DEBUG_TEXT:
            print("[GEM][text] embed candidates found:", len(candidates))
            print("[GEM][text] model type:", type(model))
            print("[GEM][text] base_model type:", type(base_model))
            if hasattr(base_model, "__dict__"):
                keys = list(base_model.__dict__.keys())
                print("[GEM][text] base_model keys (subset):", keys[:10])
        self.embed_tokens = next((emb for emb in candidates if emb is not None), None)
        if self.embed_tokens is None:
            raise RuntimeError(
                "Cannot locate embed_tokens within Qwen model "
                f"(model type={type(model)}, base type={type(base_model)})"
            )
        if DEBUG_TEXT:
            print("[GEM][text] embed_tokens layer type:", type(self.embed_tokens))

    def _tokenize(self, prompts: List[str]) -> torch.Tensor:
        if DEBUG_TEXT:
            print("[GEM][text] prompts:", prompts)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)

        sequences: List[List[int]] = []
        max_len = 0
        for idx, prompt in enumerate(prompts):
            try:
                encoded = self.tokenizer(
                    prompt,
                    add_special_tokens=True,
                    return_tensors=None,
                )
            except Exception as exc:
                print(f"[GEM][text][ERR] tokenizer single prompt failed (idx={idx}):", exc)
                raise
            if isinstance(encoded, BatchEncoding):
                ids = encoded.get("input_ids")
            elif isinstance(encoded, dict):
                ids = encoded.get("input_ids")
            else:
                ids = encoded
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            if isinstance(ids, torch.Tensor):
                ids = ids.long().tolist()
            if not isinstance(ids, list):
                raise TypeError(f"Tokenizer returned unsupported type for prompt {idx}: {type(ids)}")
            if DEBUG_TEXT:
                preview = ids[: min(10, len(ids))]
                print(f"[GEM][text] prompt {idx} tokens preview:", preview)
            sequences.append(ids)
            max_len = max(max_len, len(ids))
            if DEBUG_TEXT:
                print(f"[GEM][text] prompt {idx} token length:", len(ids))

        if DEBUG_TEXT:
            print("[GEM][text] max token length:", max_len)
        # ensure no zero-length sequences
        sanitized = []
        for idx, seq in enumerate(sequences):
            if not seq:
                seq = [pad_token_id]
                if DEBUG_TEXT:
                    print(f"[GEM][text] prompt {idx} empty tokens; inserting pad token")
            sanitized.append(seq)

        batch = _pad_batch(sanitized, pad_token_id, self.device)
        if DEBUG_TEXT:
            print("[GEM][text] batch shape after pad:", batch.shape)
            print("[GEM][text] batch row 0 preview:", batch[0][:10].tolist())
        return batch

    @torch.no_grad()
    def encode(self, prompts: List[str]) -> torch.Tensor:
        input_ids = self._tokenize(prompts)
        embeddings = self.embed_tokens(input_ids)
        eos_embeddings = embeddings[:, -1, :]
        eos_embeddings = eos_embeddings / (eos_embeddings.norm(dim=-1, keepdim=True) + 1e-6)
        if DEBUG_TEXT:
            print("[GEM][text] embeddings shape:", embeddings.shape)
            print("[GEM][text] eos_embeddings shape:", eos_embeddings.shape)
        return eos_embeddings
