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
**2D RoPE** (row / col patch indices) encodes position in attention.

Inputs may have **any** height and width as long as both are divisible by
``patch_size`` — RoPE is computed per-forward from the actual patch grid, so
every batch can have a different shape. In training we shrink images taller
than ``max_image_height`` (aspect-ratio preserving) and **never stretch** or
letterbox to a fixed canvas; variable-size samples within a batch are padded
to the batch-max (rounded up to ``patch_size``) by a collate function, and a
``pad_mask`` marks the padded pixels so padding patches are excluded from
attention / loss.

References
----------
* He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
"""

import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# =====================================================================
# Building blocks
# =====================================================================

class PatchEmbed(nn.Module):
    """Convert a grayscale image into a sequence of patch embeddings.

    No fixed canvas: the forward pass accepts **any** ``[B, C, H, W]`` as long as
    both ``H`` and ``W`` are divisible by ``patch_size`` (a ``ValueError`` is raised
    otherwise). The patch grid is derived from the input at every call.
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: ``[B, C, H, W]``. Both ``H`` and ``W`` must be divisible by ``patch_size``.

        Returns:
            tokens:  ``[B, grid_h * grid_w, embed_dim]``
            grid_h:  ``H // patch_size``
            grid_w:  ``W // patch_size``
        """
        _, _, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input H and W must be divisible by patch_size; "
                f"got (H, W)={(H, W)}, patch_size={self.patch_size}."
            )
        x = self.proj(x)              # [B, embed_dim, grid_h, grid_w]
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        x = x.flatten(2).transpose(1, 2)  # [B, grid_h*grid_w, embed_dim]
        return x, grid_h, grid_w


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.size(-1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    """Multi-head self-attention with 2D rotary position embedding (row / col in head_dim halves)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        qkv_bias: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim = embed_dim // num_heads must be even for 2D RoPE, got {self.head_dim}"
            )
        # Each axis uses half the head: RoPE on first half = row, second = col
        d_rope = self.head_dim // 2
        inv = 1.0 / (
            rope_base ** (torch.arange(0, d_rope, 2, dtype=torch.float32) / max(d_rope, 1.0))
        )
        self.register_buffer("rope_inv_freq", inv, persistent=False)
        self.rope_base = rope_base
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def _apply_1d_rope(self, t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # t: [B, n_heads, T, d_sub] with d_sub even, pos: [B, T]
        inv = self.rope_inv_freq
        p = (pos.to(dtype=inv.dtype).unsqueeze(-1) * inv).unsqueeze(1)  # [B,1,T,n_pairs]
        cos = p.cos().repeat_interleave(2, dim=-1)
        sin = p.sin().repeat_interleave(2, dim=-1)
        return t * cos + _rotate_half(t) * sin

    def _apply_2d_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_h: torch.Tensor,
        pos_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d = q.size(-1)
        hdim = d // 2
        qh, qw = q.split(hdim, dim=-1)
        kh, kw = k.split(hdim, dim=-1)
        qh, kh = self._apply_1d_rope(qh, pos_h), self._apply_1d_rope(kh, pos_h)
        qw, kw = self._apply_1d_rope(qw, pos_w), self._apply_1d_rope(kw, pos_w)
        return torch.cat((qh, qw), dim=-1), torch.cat((kh, kw), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        pos_h: torch.Tensor,
        pos_w: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self._apply_2d_rope(q, k, pos_h, pos_w)

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

    def forward(
        self,
        x: torch.Tensor,
        pos_h: torch.Tensor,
        pos_w: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pos_h, pos_w)
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
        max_image_height:  Record-only upper bound on training-time image height
                           (in pixels). Stored in the checkpoint so the
                           preprocessing can be replayed; the model itself
                           accepts **any** ``(H, W)`` per forward pass as long
                           as both are divisible by ``patch_size``.
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
        fft_loss_weight: float = 0.0,
        max_image_height: Optional[int] = None,
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
        if l1_loss_weight < 0 or l2_loss_weight < 0 or fft_loss_weight < 0:
            raise ValueError(
                f"Loss weights must be non-negative, got l1={l1_loss_weight}, "
                f"l2={l2_loss_weight}, fft={fft_loss_weight}."
            )
        if l1_loss_weight == 0.0 and l2_loss_weight == 0.0:
            raise ValueError("At least one of l1_loss_weight or l2_loss_weight must be positive.")

        # norm_pix_loss targets are per-patch z-scored (roughly [-5, +5]), which
        # a sigmoid output cannot reach. If both flags are True, the decoder
        # saturates and pre-training quality collapses. Auto-disable clipping.
        if norm_pix_loss and clip_pixel_pred:
            warnings.warn(
                "clip_pixel_pred=True is incompatible with norm_pix_loss=True: "
                "sigmoid output is confined to (0, 1) but per-patch normalised "
                "targets extend outside this range, preventing the decoder from "
                "reaching them. Disabling clip_pixel_pred for this model.",
                stacklevel=2,
            )
            clip_pixel_pred = False

        # ---- store config for serialisation / feature_extractor ----
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
        self.fft_loss_weight = fft_loss_weight
        self.max_image_height = (
            int(max_image_height) if max_image_height is not None else None
        )

        # ---- encoder ----
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ---- decoder ----
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

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
            imgs: ``[B, C, H, W]``. ``H`` and ``W`` must be divisible by ``patch_size``.
        Returns:
            ``[B, grid_h * grid_w, patch_size**2 * C]``
        """
        p = self.patch_size
        c = self.in_channels
        B, _, H, W = imgs.shape
        if H % p != 0 or W % p != 0:
            raise ValueError(
                f"patchify: (H, W)={(H, W)} must be divisible by patch_size={p}."
            )
        h, w_ = H // p, W // p
        x = imgs.reshape(B, c, h, p, w_, p)
        x = x.permute(0, 2, 4, 3, 5, 1)       # [B, h, w, p, p, C]
        x = x.reshape(B, h * w_, p * p * c)
        return x

    def unpatchify(
        self, x: torch.Tensor, grid_h: int, grid_w: int,
    ) -> torch.Tensor:
        """
        Reconstruct images from patch pixel values.

        Args:
            x:       ``[B, grid_h * grid_w, patch_size**2 * C]``
            grid_h:  number of patch rows
            grid_w:  number of patch columns
        Returns:
            ``[B, C, grid_h * patch_size, grid_w * patch_size]``
        """
        p = self.patch_size
        c = self.in_channels
        B = x.shape[0]
        x = x.reshape(B, grid_h, grid_w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)       # [B, C, h, p, w, p]
        x = x.reshape(B, c, grid_h * p, grid_w * p)
        return x

    # -----------------------------------------------------------------
    # Random masking
    # -----------------------------------------------------------------

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            ids_keep: ``[B, num_keep]``  original patch indices kept (for RoPE).
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

        return x_masked, mask, ids_restore, ids_keep

    @staticmethod
    def _patch_grid_positions(
        grid_h: int, grid_w: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Row-major (row, col) float coordinates for a ``grid_h × grid_w`` patch grid.

        Returns ``(pos_h, pos_w)`` each of shape ``[grid_h * grid_w]``.
        """
        ar = torch.arange(grid_h * grid_w, device=device)
        return (ar // grid_w).float(), (ar % grid_w).float()

    def _encoder_token_positions(
        self,
        ids_keep: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RoPE (row, col) per token: CLS at (0,0), then positions of kept patches."""
        B = ids_keep.shape[0]
        device = ids_keep.device
        patch_h, patch_w = self._patch_grid_positions(grid_h, grid_w, device)
        z = torch.zeros(B, 1, device=device, dtype=patch_h.dtype)
        ph = torch.cat([z, patch_h[ids_keep]], dim=1)
        pw = torch.cat([z, patch_w[ids_keep]], dim=1)
        return ph, pw

    def _full_token_positions(
        self,
        B: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CLS (0,0) + full ``grid_h × grid_w`` patch grid in row-major order."""
        patch_h, patch_w = self._patch_grid_positions(grid_h, grid_w, device)
        z = torch.zeros(B, 1, device=device, dtype=patch_h.dtype)
        ph = torch.cat([z, patch_h.unsqueeze(0).expand(B, -1)], dim=1)
        pw = torch.cat([z, patch_w.unsqueeze(0).expand(B, -1)], dim=1)
        return ph, pw

    # -----------------------------------------------------------------
    # Encoder
    # -----------------------------------------------------------------

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Encode only the *visible* patches (training path).

        Args:
            x: ``[B, C, H, W]`` (both sides divisible by ``patch_size``).
            pad_mask: ``[B, 1, H, W]`` boolean pixel-level mask (True = pad),
                or ``None``.

        Returns:
            latent:      ``[B, N_visible + 1, embed_dim]``  (includes CLS)
            mask:        ``[B, grid_h * grid_w]``
            ids_restore: ``[B, grid_h * grid_w]``
            grid_h:      patch rows
            grid_w:      patch cols
        """
        patch_pad_mask: Optional[torch.Tensor] = None
        if pad_mask is not None:
            patch_pad_mask = _pixel_pad_mask_to_patch(
                pad_mask, self.patch_size,
            )

        x, grid_h, grid_w = self.patch_embed(x)          # [B, N, D]

        if patch_pad_mask is not None:
            x = x * (~patch_pad_mask).unsqueeze(-1).float()

        x, mask, ids_restore, ids_keep = self.random_masking(
            x, mask_ratio, pad_mask=patch_pad_mask,
        )
        pos_h, pos_w = self._encoder_token_positions(ids_keep, grid_h, grid_w)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for blk in self.blocks:
            x = blk(x, pos_h, pos_w)
        x = self.norm(x)

        return x, mask, ids_restore, grid_h, grid_w

    # -----------------------------------------------------------------
    # Decoder
    # -----------------------------------------------------------------

    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        Decode from encoded visible patches + mask tokens.

        Args:
            x:           ``[B, 1 + N_vis, embed_dim]``
            ids_restore: ``[B, grid_h * grid_w]``
            grid_h:      patch rows
            grid_w:      patch cols

        Returns:
            pred: ``[B, grid_h * grid_w, patch_size**2 * in_channels]``. If
                  ``clip_pixel_pred``, values are in ``(0, 1)`` via sigmoid.
        """
        x = self.decoder_embed(x)                       # [B, 1+N_vis, dec_dim]

        N = grid_h * grid_w
        num_masked = N - (x.shape[1] - 1)  # -1 for CLS
        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1)

        x_no_cls = x[:, 1:, :]                          # [B, N_vis, dec_dim]
        x_ = torch.cat([x_no_cls, mask_tokens], dim=1)  # [B, N, dec_dim]
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]),
        )

        x = torch.cat([x[:, :1, :], x_], dim=1)         # [B, 1+N, dec_dim]

        B = x.shape[0]
        pos_h, pos_w = self._full_token_positions(B, grid_h, grid_w, x.device)

        for blk in self.decoder_blocks:
            x = blk(x, pos_h, pos_w)
        x = self.decoder_norm(x)

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
        target_raw = self.patchify(imgs)                   # [B, N, p*p*C] — raw pixels

        if self.norm_pix_loss:
            mean = target_raw.mean(dim=-1, keepdim=True)
            var  = target_raw.var(dim=-1, keepdim=True)
            target = (target_raw - mean) / (var + 1e-6).sqrt()
        else:
            target = target_raw

        diff = pred - target
        l1 = diff.abs().mean(dim=-1)                      # [B, N]
        l2 = (diff ** 2).mean(dim=-1)                     # [B, N]
        loss = self.l1_loss_weight * l1 + self.l2_loss_weight * l2

        if self.fft_loss_weight > 0:
            p = self.patch_size
            B, N, _ = pred.shape
            # FFT is computed on raw (un-normalized) patches so the DC component
            # (inter-patch brightness) is preserved. If norm_pix_loss is on, we
            # must un-normalize pred back to raw pixel space first.
            if self.norm_pix_loss:
                pred_raw = pred * (var + 1e-6).sqrt() + mean
            else:
                pred_raw = pred
            pred_s = pred_raw.reshape(B * N, self.in_channels, p, p)
            tgt_s  = target_raw.reshape(B * N, self.in_channels, p, p)
            # norm="ortho" makes the transform unitary (energy-preserving).
            fft_diff = (
                torch.fft.rfft2(pred_s, norm="ortho")
                - torch.fft.rfft2(tgt_s, norm="ortho")
            )
            # Mean L1 over all frequency bins and channels → [B, N]
            fft_loss = fft_diff.abs().mean(dim=(-3, -2, -1)).reshape(B, N)
            loss = loss + self.fft_loss_weight * fft_loss

        effective_mask = mask
        if pad_mask is not None:
            patch_pad = _pixel_pad_mask_to_patch(pad_mask, self.patch_size)
            effective_mask = mask * (~patch_pad).float()

        denom = effective_mask.sum().clamp_min(1.0)
        loss = (loss * effective_mask).sum() / denom
        return loss

    def reference_mse(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pure normalized per-patch MSE on masked content patches.

        Independent of l1/l2/fft loss weights — use this to compare runs
        trained with different loss configurations on equal footing.
        Matches the metric reported in He et al. (MAE, 2022).
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        mse_per_patch = (pred - target).pow(2).mean(dim=-1)   # [B, N]
        eff_mask = mask
        if pad_mask is not None:
            pp = _pixel_pad_mask_to_patch(pad_mask, self.patch_size)
            eff_mask = mask * (~pp).float()
        return (mse_per_patch * eff_mask).sum() / eff_mask.sum().clamp_min(1.0)

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
        latent, mask, ids_restore, grid_h, grid_w = self.forward_encoder(
            imgs, mask_ratio, pad_mask=pad_mask,
        )
        pred = self.forward_decoder(latent, ids_restore, grid_h, grid_w)
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
        pool: str = "mean",
    ) -> torch.Tensor:
        """
        Extract encoder embeddings by feeding *all* patches through the encoder
        (no masking).

        Args:
            imgs: ``[B, C, H, W]``
            pad_mask: ``[B, 1, H, W]`` boolean (True = pad), or ``None``.
            pool: How to reduce the encoder output to a single ``[B, embed_dim]``
                vector:

                * ``"mean"`` *(default, recommended for MAE)*: average over
                  patch tokens, excluding padding patches when ``pad_mask`` is
                  provided. Per He et al. 2022 (MAE, §A.2), mean-pooled patch
                  tokens consistently outperform the CLS token on linear
                  probe for MAE-pretrained encoders (the CLS token has no
                  direct pre-training target and is under-trained).
                * ``"cls"``: return the CLS token (kept for compatibility).

        Returns:
            ``[B, embed_dim]``
        """
        if pool not in ("mean", "cls"):
            raise ValueError(f"pool must be 'mean' or 'cls', got {pool!r}")

        x, grid_h, grid_w = self.patch_embed(imgs)       # [B, N, D]

        patch_pad: Optional[torch.Tensor] = None
        if pad_mask is not None:
            patch_pad = _pixel_pad_mask_to_patch(pad_mask, self.patch_size)
            x = x * (~patch_pad).unsqueeze(-1).float()

        B = x.shape[0]
        pos_h, pos_w = self._full_token_positions(B, grid_h, grid_w, x.device)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for blk in self.blocks:
            x = blk(x, pos_h, pos_w)
        x = self.norm(x)

        if pool == "cls":
            return x[:, 0]

        # Mean pool over patch tokens, ignoring padding.
        patches = x[:, 1:]                               # [B, N, D]
        if patch_pad is not None:
            keep = (~patch_pad).unsqueeze(-1).float()    # [B, N, 1]
            return (patches * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        return patches.mean(dim=1)

    @torch.no_grad()
    def encode_patches(
        self,
        imgs: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns [B, embed_dim, grid_h, grid_w] patch feature map for dense prediction."""
        x, grid_h, grid_w = self.patch_embed(imgs)

        if pad_mask is not None:
            patch_pad = _pixel_pad_mask_to_patch(pad_mask, self.patch_size)
            x = x * (~patch_pad).unsqueeze(-1).float()

        B = x.shape[0]
        pos_h, pos_w = self._full_token_positions(B, grid_h, grid_w, x.device)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for blk in self.blocks:
            x = blk(x, pos_h, pos_w)
        x = self.norm(x)

        patches = x[:, 1:]  # [B, N, D]
        return patches.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)

    # -----------------------------------------------------------------
    # Convenience: encoder-only model for feature extraction
    # -----------------------------------------------------------------

    def get_encoder(self) -> "MaskedAutoencoderViT":
        """Return ``self`` – the decoder is simply ignored at inference.

        Use :meth:`encode` to extract embeddings.
        """
        return self


# =====================================================================
# Factory
# =====================================================================

def create_mae_vit(
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
    fft_loss_weight: float = 0.0,
    max_image_height: Optional[int] = None,
) -> MaskedAutoencoderViT:
    """
    Create a Masked Autoencoder ViT with sensible defaults for
    grayscale ultrasound images.

    Default configuration (~11M encoder params):
        patch_size=16, embed_dim=384, depth=6, num_heads=6

    The model accepts **any** ``(H, W)`` input (both divisible by ``patch_size``).
    ``max_image_height`` is stored as metadata so the preprocessing pipeline can
    be replayed from a checkpoint.
    """
    return MaskedAutoencoderViT(
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
        fft_loss_weight=fft_loss_weight,
        max_image_height=max_image_height,
    )

