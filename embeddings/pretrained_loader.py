"""
Load pretrained ViT encoder weights into a MaskedAutoencoderViT.

Supported source: timm model names (e.g. "vit_small_patch16_224.dino",
"vit_base_patch16_224.mae", "vit_small_patch16_224").  Pass the exact
string timm.create_model() accepts.

Key adaptations:
  - patch_embed.proj.weight: 3-channel → 1-channel by averaging input filters
    (channel_avg=True, default) or by taking the first channel (channel_avg=False).
  - pos_embed: skipped — this model uses 2-D RoPE computed per-forward, there
    is no learnable absolute positional embedding.
  - blocks.{i} where i >= model.depth: skipped with a warning (allows loading
    the first N layers from a deeper pretrained model).
  - Classification head (head.*): skipped.

Example::

    from embeddings.pretrained_loader import load_pretrained_vit_encoder
    info = load_pretrained_vit_encoder(model, "vit_small_patch16_224.dino")
    print(info['summary'])
"""

import warnings
from typing import Dict, List

import torch

from embeddings.vit.model import MaskedAutoencoderViT


def load_pretrained_vit_encoder(
    model: MaskedAutoencoderViT,
    source: str,
    *,
    channel_avg: bool = True,
) -> Dict:
    """
    Load timm ViT encoder weights into *model* in-place.

    Args:
        model:       Target MaskedAutoencoderViT (already on any device).
        source:      timm model identifier, e.g. ``"vit_small_patch16_224.dino"``.
        channel_avg: How to adapt the patch-embed Conv2d from 3 in-channels to 1.
                     True  = average the 3 filters (preserves energy, recommended).
                     False = keep only the first channel's filter.

    Returns:
        Dict with keys:
            ``loaded``   – list of parameter names that were copied.
            ``skipped``  – list of (name, reason) pairs for weights that were not copied.
            ``n_loaded`` – count of loaded parameters.
            ``summary``  – human-readable one-line string.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required to load pretrained ViT weights. "
            "Install it with: pip install timm"
        ) from exc

    # ---- fetch source weights ----
    src_model = timm.create_model(source, pretrained=True)
    src_sd = src_model.state_dict()
    del src_model  # free memory immediately

    our_sd = model.state_dict()
    patch = {}    # updates to apply
    loaded: List[str] = []
    skipped: List[tuple] = []

    for src_key, src_val in src_sd.items():
        # --- absolute positional embedding ---
        if src_key == "pos_embed":
            skipped.append((src_key, "skipped: model uses 2-D RoPE, no absolute PE"))
            continue

        # --- classification head ---
        if src_key.startswith("head."):
            skipped.append((src_key, "skipped: classification head not needed"))
            continue

        # --- block depth gate ---
        if src_key.startswith("blocks."):
            block_idx = int(src_key.split(".")[1])
            if block_idx >= model.depth:
                skipped.append((
                    src_key,
                    f"skipped: pretrained block {block_idx} >= model.depth {model.depth}",
                ))
                continue

        # --- patch embed channel adaptation ---
        if src_key == "patch_embed.proj.weight":
            tgt_val = our_sd.get(src_key)
            if tgt_val is None:
                skipped.append((src_key, "skipped: not found in target model"))
                continue
            if src_val.shape[1] != tgt_val.shape[1]:
                if src_val.shape[1] == 3 and tgt_val.shape[1] == 1:
                    if channel_avg:
                        src_val = src_val.mean(dim=1, keepdim=True)
                    else:
                        src_val = src_val[:, :1]
                else:
                    skipped.append((
                        src_key,
                        f"skipped: in_channels mismatch "
                        f"src={src_val.shape[1]} tgt={tgt_val.shape[1]}",
                    ))
                    continue

        # --- generic shape check ---
        tgt_val = our_sd.get(src_key)
        if tgt_val is None:
            skipped.append((src_key, "skipped: key not found in target model"))
            continue
        if src_val.shape != tgt_val.shape:
            skipped.append((
                src_key,
                f"skipped: shape mismatch src={tuple(src_val.shape)} tgt={tuple(tgt_val.shape)}",
            ))
            continue

        patch[src_key] = src_val
        loaded.append(src_key)

    # apply
    our_sd.update(patch)
    model.load_state_dict(our_sd, strict=True)

    if model.depth > _pretrained_depth(loaded):
        warnings.warn(
            f"Only {_pretrained_depth(loaded)} blocks were available in '{source}' "
            f"but model.depth={model.depth}. The remaining blocks keep random init.",
            stacklevel=2,
        )

    summary = (
        f"pretrained '{source}' → {len(loaded)} tensors loaded, "
        f"{len(skipped)} skipped "
        f"(pos_embed/head excluded; patch-embed averaged to 1-ch)"
    )
    return {"loaded": loaded, "skipped": skipped, "n_loaded": len(loaded), "summary": summary}


def _pretrained_depth(loaded: List[str]) -> int:
    """Infer how many encoder blocks were actually loaded."""
    indices = set()
    for k in loaded:
        if k.startswith("blocks."):
            indices.add(int(k.split(".")[1]))
    return max(indices) + 1 if indices else 0
