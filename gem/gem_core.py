from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from .config import D_DYN, K_LAYERS, TAU_DYN, TAU_SELF, WS_STATIC


@torch.no_grad()
def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return F.linear(x, weight, bias)


@torch.no_grad()
def _normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def _self_self_once(projected: torch.Tensor, values: torch.Tensor, tau: float) -> torch.Tensor:
    attn_scores = torch.einsum("bnd,bmd->bnm", projected, projected) / max(1e-6, tau)
    attn_scores = attn_scores.softmax(dim=-1)
    return torch.einsum("bnm,bmd->bnd", attn_scores, values)


@torch.no_grad()
def gem_whitebox(
    layers: List[torch.Tensor],
    qkv: List[Tuple[Tuple[torch.Tensor, torch.Tensor | None], ...]],
    has_cls: bool,
    text_eos: torch.Tensor,
) -> Dict[str, Any]:
    num_layers = len(layers)
    if len(qkv) != num_layers:
        raise ValueError("qkv length mismatch")
    if len(WS_STATIC) != K_LAYERS + 1:
        raise ValueError("WS_STATIC length must equal K_LAYERS + 1")

    start_idx = max(0, num_layers - K_LAYERS)
    active_layers = list(range(start_idx, num_layers))
    base_features = layers[start_idx]
    batch, tokens, hidden = base_features.shape
    device = base_features.device

    dynamic_layer_indices = active_layers[-D_DYN:] if D_DYN > 0 else []
    dynamic_scores = []
    if has_cls and D_DYN > 0:
        cls_layers = torch.stack([layer[:, 0, :] for layer in layers])
        cls_final = cls_layers[-1].mean(dim=0)
        cls_final = _normalize(cls_final)
        eos_embedding = _normalize(text_eos.to(device).float())
        for layer_index in dynamic_layer_indices:
            if layer_index == 0:
                residual = cls_layers[layer_index]
            else:
                residual = cls_layers[layer_index] - cls_layers[layer_index - 1]
            residual = residual.mean(dim=0)
            score = F.cosine_similarity(cls_final - residual, eos_embedding, dim=0)
            dynamic_scores.append(score.item())
        if dynamic_scores:
            logits = torch.tensor(dynamic_scores, device=device)
            dynamic_weights = torch.softmax(-logits * TAU_DYN, dim=-1)
        else:
            dynamic_weights = None
    else:
        dynamic_weights = None

    static_weights = torch.tensor(WS_STATIC, device=device)
    if dynamic_weights is not None:
        combined_weights: List[torch.Tensor] = [static_weights[0]]
        for offset, layer_index in enumerate(active_layers):
            weight = static_weights[offset + 1]
            if layer_index in dynamic_layer_indices:
                dyn_idx = dynamic_layer_indices.index(layer_index)
                weight = weight - (1.0 / len(dynamic_layer_indices)) + dynamic_weights[dyn_idx]
            combined_weights.append(weight)
        combined_weights_tensor = torch.stack(combined_weights)
    else:
        combined_weights_tensor = static_weights

    fused_per_layer: List[torch.Tensor] = []
    for layer_index in active_layers:
        features = layers[layer_index]
        (w_q, b_q), (w_k, b_k), (w_v, b_v) = qkv[layer_index]
        w_q = w_q.to(device)
        w_k = w_k.to(device)
        w_v = w_v.to(device)
        b_q = b_q.to(device) if b_q is not None else None
        b_k = b_k.to(device) if b_k is not None else None
        b_v = b_v.to(device) if b_v is not None else None

        projected_q = _normalize(_linear(features, w_q, b_q))
        projected_k = _normalize(_linear(features, w_k, b_k))
        projected_v = _linear(features, w_v, b_v)

        attn_q = _self_self_once(projected_q, projected_v, TAU_SELF)
        attn_k = _self_self_once(projected_k, projected_v, TAU_SELF)
        attn_v = _self_self_once(_normalize(projected_v), projected_v, TAU_SELF)
        fused = (attn_q + attn_k + attn_v) / 3.0
        fused_per_layer.append(fused)

    fused_tokens = combined_weights_tensor[0] * base_features
    for index, per_layer_tokens in enumerate(fused_per_layer):
        fused_tokens = fused_tokens + combined_weights_tensor[index + 1] * per_layer_tokens

    return {
        "O_comb": fused_tokens,
        "w_comb": combined_weights_tensor.detach().cpu().tolist(),
    }
