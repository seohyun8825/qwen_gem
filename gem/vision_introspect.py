import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def _find_vision_tower(model: nn.Module):
    candidates = [
        getattr(model, "get_vision_tower", None),
        lambda: getattr(getattr(model, "get_model", lambda: model)(), "get_vision_tower")(),
        lambda: getattr(model, "vision_tower", None),
        lambda: getattr(getattr(model, "model", None), "vision_tower", None),
    ]
    for fn in candidates:
        try:
            tower = fn() if callable(fn) else fn
            if tower is not None:
                return getattr(tower, "vision_tower", tower)
        except Exception:
            continue
    raise RuntimeError("vision tower not found; please expose model.get_vision_tower().vision_tower")


def _collect_blocks(vit: nn.Module) -> Tuple[List[nn.Module], Any]:
    blocks = getattr(vit, "blocks", None)
    if blocks is None and hasattr(vit, "visual"):
        blocks = getattr(vit.visual, "blocks", None)
    if blocks is None and hasattr(vit, "vision_model"):
        encoder = getattr(vit.vision_model, "encoder", None)
        if encoder is not None and hasattr(encoder, "layers"):
            blocks = encoder.layers
    if blocks is None:
        blocks = [
            module
            for module in vit.modules()
            if any(hasattr(module, attr) for attr in ("attn", "attn_self", "self_attn"))
            and hasattr(module, "mlp")
        ]
    if not blocks:
        raise RuntimeError("cannot locate ViT blocks")
    patch_embed = getattr(vit, "patch_embed", None)
    if patch_embed is None and hasattr(vit, "visual"):
        patch_embed = getattr(vit.visual, "patch_embed", None)
    if patch_embed is None and hasattr(vit, "vision_model"):
        embeddings = getattr(vit.vision_model, "embeddings", None)
        if embeddings is not None:
            patch_embed = getattr(embeddings, "patch_embedding", None)
    return list(blocks), patch_embed


def _grid_from_tokens(num_tokens: int, has_cls: bool = True) -> Tuple[int, int]:
    effective_tokens = num_tokens - (1 if has_cls else 0)
    side = int(math.sqrt(effective_tokens))
    if side * side != effective_tokens:
        side = int(round(math.sqrt(effective_tokens)))
    return side, side


class ViTWhitebox:
    def __init__(self, hf_llava_model: nn.Module, device: str = "cuda"):
        self.device = device
        self.vit = _find_vision_tower(hf_llava_model).to(device)
        self.blocks, self.patch_embed = _collect_blocks(self.vit)
        self.num_layers = len(self.blocks)
        self.has_cls = True
        self.embed_dim = None
        self._layer_outs: List[torch.Tensor] = []

        def _hook(_, __, output):
            if isinstance(output, tuple):
                tensor_out = next((item for item in output if torch.is_tensor(item)), None)
                if tensor_out is None:
                    raise RuntimeError("hook received tuple without tensor outputs")
                self._layer_outs.append(tensor_out.detach())
            else:
                self._layer_outs.append(output.detach())

        self._hooks = [block.register_forward_hook(_hook) for block in self.blocks]

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                continue

    def _qkv_weights(self, block: nn.Module):
        attn = None
        for name in ("attn", "attn_self", "self_attn", "attention"):
            attn = getattr(block, name, None)
            if attn is not None:
                break
        if attn is None:
            raise RuntimeError("block has no attn")
        if hasattr(attn, "qkv"):
            weight = attn.qkv.weight
            bias = getattr(attn.qkv, "bias", None)
            dim = weight.shape[0] // 3
            w_q = weight[:dim, :]
            w_k = weight[dim : 2 * dim, :]
            w_v = weight[2 * dim :, :]
            b_q = bias[:dim] if bias is not None else None
            b_k = bias[dim : 2 * dim] if bias is not None else None
            b_v = bias[2 * dim :] if bias is not None else None
            return (w_q, b_q), (w_k, b_k), (w_v, b_v)

        def _get_linear(name_options):
            for name in name_options:
                if hasattr(attn, name):
                    return getattr(attn, name)
            return None

        proj_q = _get_linear(["q_proj", "q", "q_linear", "to_q"])
        proj_k = _get_linear(["k_proj", "k", "k_linear", "to_k"])
        proj_v = _get_linear(["v_proj", "v", "v_linear", "to_v"])
        if proj_q is None or proj_k is None or proj_v is None:
            raise RuntimeError("cannot find q/k/v projections")
        return (
            (proj_q.weight, getattr(proj_q, "bias", None)),
            (proj_k.weight, getattr(proj_k, "bias", None)),
            (proj_v.weight, getattr(proj_v, "bias", None)),
        )

    @torch.no_grad()
    def encode_frames(self, pixel_values: torch.Tensor) -> Dict[str, Any]:
        self._layer_outs.clear()
        _ = self.vit(pixel_values.to(self.device))
        if not self._layer_outs:
            raise RuntimeError("no layer outputs captured; check hooks/forward path")

        layers: List[torch.Tensor] = []
        for output in self._layer_outs:
            if output.dim() == 3:
                tensor = output
            elif output.dim() == 4:
                tensor = output.flatten(2).transpose(1, 2)
            else:
                tensor = output
            layers.append(tensor)

        temporal, tokens, embed_dim = layers[-1].shape
        self.embed_dim = embed_dim
        grid_h, grid_w = _grid_from_tokens(tokens, has_cls=True)
        if grid_h * grid_w + 1 != tokens:
            grid_h, grid_w = _grid_from_tokens(tokens, has_cls=False)
            self.has_cls = False

        if self.has_cls:
            cls_per_layer = torch.stack([layer[:, 0, :] for layer in layers])
        else:
            cls_per_layer = torch.stack([layer[:, :0, :] for layer in layers])

        qkv = [self._qkv_weights(block) for block in self.blocks]

        return {
            "layers": layers,
            "cls_per_layer": cls_per_layer,
            "grid": (grid_h, grid_w),
            "has_cls": self.has_cls,
            "qkv": qkv,
            "embed_dim": embed_dim,
        }
