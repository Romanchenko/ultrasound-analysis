"""
Custom Vision Transformer with Masked Autoencoder (MAE) for
self-supervised learning on grayscale ultrasound images.

Architecture
------------
* **Encoder** – standard ViT operating only on *visible* (unmasked) patches
  during training, or on all patches during inference / embedding extraction.
* **Decoder** – lightweight transformer that takes the encoded visible patches
  together with learnable mask tokens and reconstructs the original pixel values
  of the masked patches.

The encoder accepts **single-channel (grayscale)** input natively via a
``Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)`` projection.

References
----------
* He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
"""

import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# =====================================================================
# Building blocks
# =====================================================================

class PatchEmbed(nn.Module):
    """Convert a grayscale image into a sequence of patch embeddings."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, num_patches, embed_dim]
        """
        x = self.proj(x)              # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # print(f"Attention x shape. B: {B}, N: {N}, C: {C}")
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _make_decoder_pred_head(
    decoder_embed_dim: int,
    patch_pixels: int,
    num_layers: int,
    hidden_dim: int,
) -> nn.Module:
    """
    Map decoder token features to per-patch pixel logits.

    * ``num_layers == 1``: single linear (original MAE head).
    * ``num_layers >= 2``: stack of ``num_layers`` linear layers with GELU between them
      (last layer has no activation).
    """
    if num_layers < 1:
        raise ValueError(f"decoder_pred_num_layers must be >= 1, got {num_layers}")
    if num_layers == 1:
        return nn.Linear(decoder_embed_dim, patch_pixels, bias=True)
    layers: List[nn.Module] = []
    d_in = decoder_embed_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(d_in, hidden_dim))
        layers.append(nn.GELU())
        d_in = hidden_dim
    layers.append(nn.Linear(d_in, patch_pixels))
    return nn.Sequential(*layers)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (LayerNorm → Attention / MLP)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =====================================================================
# Masked Autoencoder ViT
# =====================================================================

def _pixel_pad_mask_to_patch(
    pad_mask: torch.Tensor,
    patch_size: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert a pixel-level padding mask to a patch-level boolean mask.

    Args:
        pad_mask: ``[B, 1, H, W]`` boolean (True = padding pixel).
        patch_size: Side length of each patch.
        threshold: A patch is considered "pad" if more than this fraction
            of its pixels are padding.

    Returns:
        ``[B, N]`` boolean, True where the patch is a padding patch.
    """
    B, _, H, W = pad_mask.shape
    h, w = H // patch_size, W // patch_size
    # Reshape to [B, h, patch_size, w, patch_size] then compute fraction
    pm = pad_mask[:, 0].reshape(B, h, patch_size, w, patch_size)
    frac = pm.float().mean(dim=(2, 4))  # [B, h, w]
    return (frac > threshold).reshape(B, h * w)  # [B, N]


class MaskedAutoencoderViT(nn.Module):
    """
    Vision Transformer with a Masked Autoencoder (MAE) head for
    self-supervised pre-training on **grayscale** images.

    During training the encoder processes only the *visible* patches
    (saving compute), while the decoder reconstructs the masked ones.

    For inference / embedding extraction call :meth:`encode` which
    feeds **all** patches through the encoder and returns the CLS-token
    representation.

    Args:
        image_size:        Input image spatial size (square).
        patch_size:        Side length of each patch.
        in_channels:       Number of input channels (1 for grayscale).
        embed_dim:         Encoder embedding dimension.
        depth:             Number of encoder transformer blocks.
        num_heads:         Number of attention heads in the encoder.
        decoder_embed_dim: Decoder embedding dimension.
        decoder_depth:     Number of decoder transformer blocks.
        decoder_num_heads: Number of attention heads in the decoder.
        mlp_ratio:         MLP hidden-dim / embed-dim ratio.
        norm_pix_loss:     If True, normalise each patch to zero-mean
                           unit-variance before computing reconstruction loss.
        clip_pixel_pred:   If True, pass decoder logits through :func:`torch.sigmoid` so
                           pixel predictions lie in ``(0, 1)``, matching ``ToTensor()`` range
                           without hard clipping (smooth gradients; avoids zero grads from
                           :func:`torch.clamp` outside the box).
        decoder_pred_num_layers: Linear layers in the pixel head after decoder blocks
            (1 = single linear, MAE paper default; 2+ = MLP with GELU between layers).
        decoder_pred_hidden_dim: Hidden width for intermediate layers when
            ``decoder_pred_num_layers > 1``. Defaults to ``decoder_embed_dim``.
        l1_loss_weight:  Non-negative weight on mean **absolute** error per patch (L1 component).
        l2_loss_weight:  Non-negative weight on mean **squared** error per patch (L2 component).
            Combined patch loss is ``l1_loss_weight * L1_patch + l2_loss_weight * L2_patch``,
            then averaged over masked patches. Default ``(0, 1)`` recovers pure MSE.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 3,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = True,
        clip_pixel_pred: bool = True,
        decoder_pred_num_layers: int = 1,
        decoder_pred_hidden_dim: Optional[int] = None,
        l1_loss_weight: float = 0.0,
        l2_loss_weight: float = 1.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}). "
                f"Multi-head attention requires head_dim = embed_dim // num_heads to be integer."
            )
        if decoder_embed_dim % decoder_num_heads != 0:
            raise ValueError(
                f"decoder_embed_dim ({decoder_embed_dim}) must be divisible by "
                f"decoder_num_heads ({decoder_num_heads})."
            )
        if l1_loss_weight < 0 or l2_loss_weight < 0:
            raise ValueError(
                f"l1_loss_weight and l2_loss_weight must be non-negative, "
                f"got l1_loss_weight={l1_loss_weight}, l2_loss_weight={l2_loss_weight}."
            )
        if l1_loss_weight == 0.0 and l2_loss_weight == 0.0:
            raise ValueError("At least one of l1_loss_weight or l2_loss_weight must be positive.")

        # ---- store config for serialisation / feature_extractor ----
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_pix_loss = norm_pix_loss
        self.clip_pixel_pred = clip_pixel_pred
        self.decoder_pred_num_layers = decoder_pred_num_layers
        pred_hid = (
            decoder_pred_hidden_dim
            if decoder_pred_hidden_dim is not None
            else decoder_embed_dim
        )
        self.decoder_pred_hidden_dim = pred_hid
        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight

        # ---- encoder ----
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +1 for CLS token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ---- decoder ----
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # +1 for CLS
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        patch_pixels = patch_size ** 2 * in_channels
        self.decoder_pred = _make_decoder_pred_head(
            decoder_embed_dim,
            patch_pixels,
            decoder_pred_num_layers,
            pred_hid,
        )

        self._init_weights()

    # -----------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------

    def _init_weights(self):
        # Sinusoidal positional embeddings (fixed, not learned)
        _init_pos_embed(self.pos_embed, self.patch_embed.num_patches, self.embed_dim)
        _init_pos_embed(
            self.decoder_pos_embed,
            self.patch_embed.num_patches,
            self.decoder_embed_dim,
        )

        # Patch projection like a linear layer
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.size(0), -1))

        # Tokens
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # Linear layers & LayerNorms
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------
    # Patchify / unpatchify helpers
    # -----------------------------------------------------------------

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patch pixel values.

        Args:
            imgs: [B, C, H, W]
        Returns:
            [B, num_patches, patch_size**2 * C]
        """
        p = self.patch_size
        c = self.in_channels
        h = w = self.image_size // p
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)       # [B, h, w, p, p, C]
        x = x.reshape(imgs.shape[0], h * w, p * p * c)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images from patch pixel values.

        Args:
            x: [B, num_patches, patch_size**2 * C]
        Returns:
            [B, C, H, W]
        """
        p = self.patch_size
        c = self.in_channels
        h = w = self.image_size // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)       # [B, C, h, p, w, p]
        x = x.reshape(x.shape[0], c, h * p, w * p)
        return x

    # -----------------------------------------------------------------
    # Random masking
    # -----------------------------------------------------------------

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Per-sample random masking by shuffling patch indices.

        Args:
            x: ``[B, N, D]``  (patch embeddings, no CLS)
            mask_ratio: fraction of **content** patches to mask.
            pad_mask: ``[B, N]`` boolean, True = padding patch.  Padding
                patches are always forced into the masked (removed) set
                and ``num_keep`` is computed relative to content patches
                only.

        Returns:
            x_masked: ``[B, num_keep, D]``
            mask:     ``[B, N]``  binary, 1 = masked (removed), 0 = kept
            ids_restore: ``[B, N]``  indices to unshuffle
        """
        B, N, D = x.shape

        noise = torch.rand(B, N, device=x.device)

        if pad_mask is not None:
            # Force padding patches to sort last (always masked)
            noise = noise.masked_fill(pad_mask, float('inf'))
            n_content = (~pad_mask).float().sum(dim=1)  # [B]
            num_keep = (n_content * (1 - mask_ratio)).int().min().item()
            num_keep = max(num_keep, 1)
        else:
            num_keep = int(N * (1 - mask_ratio))

        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    # -----------------------------------------------------------------
    # Encoder
    # -----------------------------------------------------------------

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode only the *visible* patches (training path).

        Args:
            x: ``[B, C, H, W]``
            pad_mask: ``[B, 1, H, W]`` boolean pixel-level mask (True = pad),
                or ``None``.

        Returns:
            latent:      ``[B, N_visible + 1, embed_dim]``  (includes CLS)
            mask:        ``[B, N]``
            ids_restore: ``[B, N]``
        """
        patch_pad_mask: Optional[torch.Tensor] = None
        if pad_mask is not None:
            patch_pad_mask = _pixel_pad_mask_to_patch(
                pad_mask, self.patch_size,
            )

        x = self.patch_embed(x)                          # [B, N, D]
        x = x + self.pos_embed[:, 1:, :]

        # Zero-out padding patch embeddings before masking
        if patch_pad_mask is not None:
            x = x * (~patch_pad_mask).unsqueeze(-1).float()

        x, mask, ids_restore = self.random_masking(
            x, mask_ratio, pad_mask=patch_pad_mask,
        )

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    # -----------------------------------------------------------------
    # Decoder
    # -----------------------------------------------------------------

    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode from encoded visible patches + mask tokens.

        Args:
            x:           [B, 1 + N_vis, embed_dim]
            ids_restore: [B, N]

        Returns:
            pred: [B, N, patch_size**2 * in_channels]. If ``clip_pixel_pred``, values are
                  in ``(0, 1)`` via sigmoid (logits from the pixel prediction head).
        """
        # Project to decoder dim
        x = self.decoder_embed(x)                       # [B, 1+N_vis, dec_dim]

        # Append mask tokens for removed patches
        N = self.patch_embed.num_patches
        num_masked = N - (x.shape[1] - 1)  # -1 for CLS
        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1)

        # Concat visible (without CLS) + mask tokens, then unshuffle
        x_no_cls = x[:, 1:, :]                          # [B, N_vis, dec_dim]
        x_ = torch.cat([x_no_cls, mask_tokens], dim=1)  # [B, N, dec_dim]
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]),
        )                                                 # unshuffle

        # Re-prepend CLS
        x = torch.cat([x[:, :1, :], x_], dim=1)         # [B, 1+N, dec_dim]

        # Add decoder positional embed
        x = x + self.decoder_pos_embed

        # Decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Pixel logits → (0, 1) via sigmoid (smooth, same range as ToTensor inputs)
        x = self.decoder_pred(x[:, 1:, :])               # [B, N, p*p*C]
        if self.clip_pixel_pred:
            x = torch.sigmoid(x)

        return x

    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------

    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Weighted L1 + L2 reconstruction loss on masked **content** patches.

        Args:
            imgs: ``[B, C, H, W]``
            pred: ``[B, N, p*p*C]``
            mask: ``[B, N]``  (1 = masked)
            pad_mask: ``[B, 1, H, W]`` boolean pixel mask (True = pad),
                or ``None``.

        Returns:
            Scalar loss.
        """
        target = self.patchify(imgs)                      # [B, N, p*p*C]

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        diff = pred - target
        l1 = diff.abs().mean(dim=-1)                      # [B, N]
        l2 = (diff ** 2).mean(dim=-1)                     # [B, N]
        loss = self.l1_loss_weight * l1 + self.l2_loss_weight * l2

        effective_mask = mask
        if pad_mask is not None:
            patch_pad = _pixel_pad_mask_to_patch(pad_mask, self.patch_size)
            effective_mask = mask * (~patch_pad).float()

        denom = effective_mask.sum().clamp_min(1.0)
        loss = (loss * effective_mask).sum() / denom
        return loss

    # -----------------------------------------------------------------
    # Full forward (training)
    # -----------------------------------------------------------------

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full MAE forward pass (training).

        Args:
            imgs: ``[B, C, H, W]``
            pad_mask: ``[B, 1, H, W]`` boolean (True = pad), or ``None``.

        Returns:
            loss:  scalar reconstruction loss
            pred:  ``[B, N, p*p*C]``
            mask:  ``[B, N]``
        """
        latent, mask, ids_restore = self.forward_encoder(
            imgs, mask_ratio, pad_mask=pad_mask,
        )
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask, pad_mask=pad_mask)
        return loss, pred, mask

    # -----------------------------------------------------------------
    # Embedding extraction (inference)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        imgs: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract CLS-token embeddings by feeding *all* patches through
        the encoder (no masking).

        Args:
            imgs: ``[B, C, H, W]``
            pad_mask: ``[B, 1, H, W]`` boolean (True = pad), or ``None``.

        Returns:
            ``[B, embed_dim]``
        """
        x = self.patch_embed(imgs)                       # [B, N, D]
        x = x + self.pos_embed[:, 1:, :]

        if pad_mask is not None:
            patch_pad = _pixel_pad_mask_to_patch(pad_mask, self.patch_size)
            x = x * (~patch_pad).unsqueeze(-1).float()

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x[:, 0]                                    # CLS token

    # -----------------------------------------------------------------
    # Convenience: encoder-only model for feature extraction
    # -----------------------------------------------------------------

    def get_encoder(self) -> "MaskedAutoencoderViT":
        """Return ``self`` – the decoder is simply ignored at inference.

        Use :meth:`encode` to extract embeddings.
        """
        return self


# =====================================================================
# Sinusoidal position embedding initialisation
# =====================================================================

def _init_pos_embed(
    pos_embed: nn.Parameter,
    num_patches: int,
    embed_dim: int,
):
    """Fill *pos_embed* with 2-D sinusoidal values (fixed, not learned)."""
    grid_size = int(num_patches ** 0.5)
    assert grid_size * grid_size == num_patches

    pe = _get_2d_sincos_pos_embed(embed_dim, grid_size)   # [N, D]
    # Prepend a zero row for CLS
    pe = torch.cat([torch.zeros(1, embed_dim), pe], dim=0)  # [1+N, D]
    pos_embed.data.copy_(pe.unsqueeze(0))


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    Generate 2-D sine-cosine positional embeddings.

    Returns:
        [grid_size*grid_size, embed_dim]
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing='ij'), dim=0)
    grid = grid.reshape(2, -1).T                          # [N, 2]

    half_dim = embed_dim // 2
    pe_h = _get_1d_sincos(grid[:, 0], half_dim)            # [N, D/2]
    pe_w = _get_1d_sincos(grid[:, 1], half_dim)            # [N, D/2]

    return torch.cat([pe_h, pe_w], dim=1)                  # [N, D]


def _get_1d_sincos(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sine-cosine embedding for a 1-D position sequence.

    Args:
        positions: [N]
        dim:       output dimension (must be even)

    Returns:
        [N, dim]
    """
    assert dim % 2 == 0
    omega = torch.arange(dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / (dim // 2)))

    out = positions.unsqueeze(1) * omega.unsqueeze(0)       # [N, D/2]
    return torch.cat([out.sin(), out.cos()], dim=1)          # [N, D]


# =====================================================================
# Factory
# =====================================================================

def create_mae_vit(
    image_size: int = 224,
    patch_size: int = 16,
    in_channels: int = 1,
    embed_dim: int = 384,
    depth: int = 6,
    num_heads: int = 6,
    decoder_embed_dim: int = 192,
    decoder_depth: int = 4,
    decoder_num_heads: int = 3,
    mlp_ratio: float = 4.0,
    norm_pix_loss: bool = True,
    clip_pixel_pred: bool = True,
    decoder_pred_num_layers: int = 1,
    decoder_pred_hidden_dim: Optional[int] = None,
    l1_loss_weight: float = 0.0,
    l2_loss_weight: float = 1.0,
) -> MaskedAutoencoderViT:
    """
    Create a Masked Autoencoder ViT with sensible defaults for
    grayscale ultrasound images.

    Default configuration (~11M encoder params):
        patch_size=16, embed_dim=384, depth=6, num_heads=6
    """
    return MaskedAutoencoderViT(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=mlp_ratio,
        norm_pix_loss=norm_pix_loss,
        clip_pixel_pred=clip_pixel_pred,
        decoder_pred_num_layers=decoder_pred_num_layers,
        decoder_pred_hidden_dim=decoder_pred_hidden_dim,
        l1_loss_weight=l1_loss_weight,
        l2_loss_weight=l2_loss_weight,
    )

