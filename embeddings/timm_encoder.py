"""
Drop-in encoder wrapper around a pretrained timm ViT.

Exposes the same interface as MaskedAutoencoderViT so it can be used
anywhere an encoder is expected (linear probe scripts, VICReg, RankME, etc.):

    encoder.encode(imgs, pad_mask=None)  →  [B, embed_dim]
    encoder.embed_dim
    encoder.patch_size

Grayscale adaptation: the RGB patch-embed Conv2d is replaced with a 1-channel
version whose weights are averaged over the three input filters (channel_avg=True,
default) or taken from the first channel only (channel_avg=False).

Variable-size inputs: pass ``dynamic_img_size=True`` (the default) to timm, which
bilinearly interpolates positional embeddings on the fly so any (H, W) divisible
by patch_size is accepted.

Example::

    from embeddings.timm_encoder import TimmViTEncoder
    enc = TimmViTEncoder("vit_small_patch16_224.dino")
    emb = enc.encode(torch.randn(4, 1, 256, 320))   # [4, 384]
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimmViTEncoder(nn.Module):
    """
    Wraps a pretrained timm ViT as a frozen or fine-tunable 1-channel encoder.

    Args:
        model_name:    timm model identifier, e.g. ``"vit_small_patch16_224.dino"``.
        pretrained:    Download pretrained weights. Default True.
        channel_avg:   How to convert 3-channel patch embed → 1-channel.
                       True = average the three filters (recommended).
                       False = keep only the first channel's filter.
        dynamic_img_size: Allow inputs with sizes other than the model's native
                       training size. Positional embeddings are interpolated.
                       Default True.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        *,
        channel_avg: bool = True,
        dynamic_img_size: bool = True,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required. Install with: pip install timm"
            ) from exc

        try:
            self._timm = timm.create_model(
                model_name, pretrained=pretrained, dynamic_img_size=dynamic_img_size,
            )
        except TypeError:
            self._timm = timm.create_model(model_name, pretrained=pretrained)

        self._adapt_patch_embed(channel_avg)

        self.model_name: str = model_name
        self.embed_dim: int = self._timm.embed_dim
        ps = self._timm.patch_embed.patch_size
        self.patch_size: int = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
        # num_prefix_tokens = CLS + register tokens (1 for standard ViT, 5 for reg4, etc.)
        self._n_prefix: int = getattr(self._timm, 'num_prefix_tokens', 1)
        self.max_image_height: Optional[int] = None  # no fixed cap; variable size is supported

    def _adapt_patch_embed(self, channel_avg: bool) -> None:
        """Replace 3-channel Conv2d with a 1-channel equivalent."""
        proj = self._timm.patch_embed.proj
        if proj.in_channels == 1:
            return
        w = proj.weight.data                              # [out, 3, kH, kW]
        new_w = w.mean(dim=1, keepdim=True) if channel_avg else w[:, :1]
        has_bias = proj.bias is not None
        new_proj = nn.Conv2d(
            1, proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=has_bias,
        )
        new_proj.weight.data = new_w
        if has_bias:
            new_proj.bias.data = proj.bias.data.clone()
        self._timm.patch_embed.proj = new_proj

    def _init_weights(self) -> None:
        """Re-initialise all weights (random encoder baseline)."""
        if hasattr(self._timm, 'init_weights'):
            self._timm.init_weights()
        else:
            for m in self._timm.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def encode(
        self,
        imgs: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        pool: str = "mean",
    ) -> torch.Tensor:
        """
        Extract encoder embeddings.

        Args:
            imgs:     ``[B, 1, H, W]`` grayscale, values in any range (model will
                      process them as-is — apply standardisation upstream if needed).
            pad_mask: ``[B, 1, H, W]`` boolean, True = padding pixel. Used to
                      exclude padded patches from mean pooling. Ignored when
                      ``pool="cls"``.
            pool:     ``"mean"`` (default) — average over patch tokens, excluding
                      padding. ``"cls"`` — return the CLS token.

        Returns:
            ``[B, embed_dim]``
        """
        if pool not in ("mean", "cls"):
            raise ValueError(f"pool must be 'mean' or 'cls', got {pool!r}")

        x = self._timm.forward_features(imgs)  # [B, n_prefix+N, D]

        if pool == "cls":
            return x[:, 0]

        patch_tokens = x[:, self._n_prefix:]  # [B, N, D] — skip CLS + register tokens

        if pad_mask is not None:
            B, _, H, W = pad_mask.shape
            ps = self.patch_size
            gh, gw = H // ps, W // ps
            pm = pad_mask[:, 0].reshape(B, gh, ps, gw, ps).float().mean(dim=(2, 4))
            is_pad = (pm > 0.5).reshape(B, gh * gw)     # [B, N]
            keep = (~is_pad).unsqueeze(-1).float()       # [B, N, 1]
            return (patch_tokens * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)

        return patch_tokens.mean(dim=1)

    @torch.no_grad()
    def encode_patches(
        self,
        imgs: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns [B, embed_dim, grid_h, grid_w] patch feature map for dense prediction."""
        x = self._timm.forward_features(imgs)   # [B, n_prefix+N, D]
        patch_tokens = x[:, self._n_prefix:]    # [B, N, D]
        B, _, H, W = imgs.shape
        grid_h, grid_w = H // self.patch_size, W // self.patch_size
        return patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)

    def forward(
        self, imgs: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encode(imgs, pad_mask=pad_mask)
