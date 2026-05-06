"""
CNN + Vision Transformer with Masked Autoencoder (MAE) for
self-supervised learning on grayscale ultrasound images.

Architecture
------------
* **CNN Feature Extractor** – configurable multi-layer CNN that processes raw
  images into feature maps before ViT tokenisation.  Each layer has a
  configurable kernel size, stride and output-channel count, followed by
  BatchNorm2d and GELU.  When strides > 1 the CNN reduces spatial resolution so
  each ViT token covers a larger pixel receptive field.
* **Patch Embedding** – applied to CNN feature maps rather than raw pixels.
  The ViT ``patch_size`` governs the patch grid over the *CNN feature map*;
  the region of the original image that one token represents is
  ``patch_size × cnn_effective_stride`` pixels on each side.
* **Encoder** – standard ViT with 2D RoPE, operating only on *visible*
  (unmasked) tokens during training; all tokens during inference.
* **Decoder** – lightweight transformer reconstructing original pixel values of
  the masked patches (pixel patch side = ``patch_size × cnn_effective_stride``).

Key variable: ``pixel_patch_size = patch_size * cnn_effective_stride``
  where ``cnn_effective_stride`` is the product of all CNN layer strides.
  This governs patchify / unpatchify / the decoder prediction head /
  FFT loss / pad-mask conversion.

References
----------
* He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
"""

import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# CNN Feature Extractor
# =====================================================================

class CNNFeatureExtractor(nn.Module):
    """
    Configurable multi-layer CNN that extracts feature maps from raw images
    before ViT patch tokenisation.

    Each layer applies ``Conv2d → BatchNorm2d → GELU``.  The output feature
    map is passed directly to :class:`PatchEmbed`.

    Args:
        in_channels:    Input channels (1 for grayscale).
        layer_channels: Output channels per layer.  ``len(layer_channels)``
                        sets the number of convolutional layers.
        kernel_sizes:   Kernel size(s).  A single int applies to every layer;
                        a list specifies sizes per layer.  Odd values keep
                        spatial dimensions clean with ``padding = k // 2``.
        strides:        Stride(s).  A single int applies to every layer;
                        a list specifies strides per layer.
                        ``effective_stride`` equals the product of all strides.
    """

    def __init__(
        self,
        in_channels: int = 1,
        layer_channels: Optional[List[int]] = None,
        kernel_sizes: Union[int, List[int]] = 3,
        strides: Union[int, List[int]] = 1,
    ):
        super().__init__()
        if layer_channels is None:
            layer_channels = [32, 64]
        n = len(layer_channels)
        if n == 0:
            raise ValueError("layer_channels must have at least one entry.")

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n
        if isinstance(strides, int):
            strides = [strides] * n

        if len(kernel_sizes) != n:
            raise ValueError(
                f"kernel_sizes length ({len(kernel_sizes)}) must equal "
                f"len(layer_channels) ({n})."
            )
        if len(strides) != n:
            raise ValueError(
                f"strides length ({len(strides)}) must equal "
                f"len(layer_channels) ({n})."
            )

        self.layer_channels: List[int] = list(layer_channels)
        self.kernel_sizes: List[int] = list(kernel_sizes)
        self.strides: List[int] = list(strides)

        effective_stride = 1
        for s in strides:
            effective_stride *= s
        self.effective_stride: int = effective_stride
        self.out_channels: int = layer_channels[-1]

        layers: List[nn.Module] = []
        c_in = in_channels
        for c_out, k, s in zip(layer_channels, kernel_sizes, strides):
            layers.append(
                nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=k // 2, bias=False)
            )
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.GELU())
            c_in = c_out
        self.layers_seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, in_channels, H, W]``

        Returns:
            ``[B, out_channels, H // effective_stride, W // effective_stride]``
            (exact when H and W are divisible by ``effective_stride``).
        """
        return self.layers_seq(x)


# =====================================================================
# Building blocks (identical to embeddings.vit.model)
# =====================================================================

class PatchEmbed(nn.Module):
    """Convert a feature map (or raw image) into a sequence of patch embeddings.

    No fixed canvas: accepts any ``[B, C, H, W]`` with ``H`` and ``W`` divisible
    by ``patch_size``.
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
        _, _, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input H and W must be divisible by patch_size; "
                f"got (H, W)={(H, W)}, patch_size={self.patch_size}."
            )
        x = self.proj(x)
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        x = x.flatten(2).transpose(1, 2)
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
        inv = self.rope_inv_freq
        p = (pos.to(dtype=inv.dtype).unsqueeze(-1) * inv).unsqueeze(1)
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
# Shared helpers
# =====================================================================

def _pixel_pad_mask_to_patch(
    pad_mask: torch.Tensor,
    patch_size: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert a pixel-level padding mask to a patch-level boolean mask.

    Args:
        pad_mask:   ``[B, 1, H, W]`` boolean (True = padding pixel).
        patch_size: Side length of each patch (in pixels).
        threshold:  A patch is considered "pad" if more than this fraction
                    of its pixels are padding.

    Returns:
        ``[B, N]`` boolean, True where the patch is a padding patch.
    """
    B, _, H, W = pad_mask.shape
    h, w = H // patch_size, W // patch_size
    pm = pad_mask[:, 0].reshape(B, h, patch_size, w, patch_size)
    frac = pm.float().mean(dim=(2, 4))
    return (frac > threshold).reshape(B, h * w)


# =====================================================================
# CNN Masked Autoencoder ViT
# =====================================================================

class CNNMaskedAutoencoderViT(nn.Module):
    """
    CNN + Vision Transformer with a Masked Autoencoder (MAE) head for
    self-supervised pre-training on **grayscale** images.

    A configurable multi-layer :class:`CNNFeatureExtractor` sits between the
    raw image and the ViT patch embedding.  CNN feature maps are tokenised by
    :class:`PatchEmbed`; all subsequent transformer logic is identical to the
    plain MAE ViT.

    Pixel reconstruction (loss and visualisation) operates at
    ``pixel_patch_size = patch_size * cnn_effective_stride`` pixels per patch
    side, matching the receptive field that each token sees in the original
    image.

    Args:
        max_image_height:  Record-only upper bound on training-time image height.
                           Images must be divisible by
                           ``pixel_patch_size = patch_size * cnn_effective_stride``.
        patch_size:        ViT patch side length (applied to CNN feature maps).
        in_channels:       Original image channels (1 for grayscale).
        embed_dim:         Encoder embedding dimension.
        depth:             Number of encoder transformer blocks.
        num_heads:         Encoder attention heads.
        decoder_embed_dim: Decoder embedding dimension.
        decoder_depth:     Decoder transformer blocks.
        decoder_num_heads: Decoder attention heads.
        mlp_ratio:         MLP hidden / embed ratio.
        norm_pix_loss:     Per-patch z-score normalisation in loss.
        clip_pixel_pred:   Apply sigmoid to decoder logits (incompatible with
                           ``norm_pix_loss=True``).
        decoder_pred_num_layers: Linear layers in pixel prediction head.
        decoder_pred_hidden_dim: Hidden width for multi-layer head.
        l1_loss_weight:    Weight on L1 component of reconstruction loss.
        l2_loss_weight:    Weight on L2 component.
        fft_loss_weight:   Weight on frequency-domain loss.
        cnn_layer_channels: Output channels per CNN layer.  Length sets the
                            number of convolutional layers.
        cnn_kernel_sizes:  Kernel size(s) for CNN layers.
        cnn_strides:       Stride(s) for CNN layers.
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
        # CNN config
        cnn_layer_channels: Optional[List[int]] = None,
        cnn_kernel_sizes: Union[int, List[int]] = 3,
        cnn_strides: Union[int, List[int]] = 1,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
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

        if norm_pix_loss and clip_pixel_pred:
            warnings.warn(
                "clip_pixel_pred=True is incompatible with norm_pix_loss=True: "
                "sigmoid output is confined to (0, 1) but per-patch normalised "
                "targets extend outside this range. Disabling clip_pixel_pred.",
                stacklevel=2,
            )
            clip_pixel_pred = False

        # ---- CNN feature extractor ----
        if cnn_layer_channels is not None and len(cnn_layer_channels) > 0:
            self.cnn: Optional[CNNFeatureExtractor] = CNNFeatureExtractor(
                in_channels=in_channels,
                layer_channels=cnn_layer_channels,
                kernel_sizes=cnn_kernel_sizes,
                strides=cnn_strides,
            )
            patch_embed_in_ch = self.cnn.out_channels
            cnn_effective_stride = self.cnn.effective_stride
        else:
            self.cnn = None
            patch_embed_in_ch = in_channels
            cnn_effective_stride = 1
            # Normalise to lists for serialisation even when CNN is absent
            if isinstance(cnn_kernel_sizes, int):
                cnn_kernel_sizes = []
            if isinstance(cnn_strides, int):
                cnn_strides = []

        # pixel_patch_size: how many original pixels each token covers (per side)
        self.pixel_patch_size: int = patch_size * cnn_effective_stride
        self.cnn_effective_stride: int = cnn_effective_stride

        # ---- store config ----
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
            decoder_pred_hidden_dim if decoder_pred_hidden_dim is not None
            else decoder_embed_dim
        )
        self.decoder_pred_hidden_dim = pred_hid
        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.fft_loss_weight = fft_loss_weight
        self.max_image_height = int(max_image_height) if max_image_height is not None else None
        self.cnn_layer_channels: Optional[List[int]] = (
            list(cnn_layer_channels) if cnn_layer_channels else None
        )
        self.cnn_kernel_sizes: List[int] = (
            self.cnn.kernel_sizes if self.cnn is not None else []
        )
        self.cnn_strides: List[int] = (
            self.cnn.strides if self.cnn is not None else []
        )

        # ---- encoder ----
        self.patch_embed = PatchEmbed(patch_size, patch_embed_in_ch, embed_dim)
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
        # decoder predicts original pixel values at pixel_patch_size resolution
        patch_pixels = self.pixel_patch_size ** 2 * in_channels
        self.decoder_pred = _make_decoder_pred_head(
            decoder_embed_dim, patch_pixels, decoder_pred_num_layers, pred_hid,
        )

        self._init_weights()

    # -----------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        # Linear / LayerNorm / CNN Conv2d
        self.apply(self._init_module)
        # Override patch_embed.proj with xavier (better for ViT projection)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.size(0), -1))

    @staticmethod
    def _init_module(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------
    # Patchify / unpatchify (operate on original pixel space)
    # -----------------------------------------------------------------

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patch pixel values.

        Uses ``pixel_patch_size = patch_size * cnn_effective_stride`` so each
        patch corresponds to the full original-pixel receptive field of one token.

        Args:
            imgs: ``[B, C, H, W]``. Both sides must be divisible by
                  ``pixel_patch_size``.
        Returns:
            ``[B, N, pixel_patch_size**2 * C]``
        """
        p = self.pixel_patch_size
        c = self.in_channels
        B, _, H, W = imgs.shape
        if H % p != 0 or W % p != 0:
            raise ValueError(
                f"patchify: (H, W)={(H, W)} must be divisible by "
                f"pixel_patch_size={p} (patch_size={self.patch_size} × "
                f"cnn_effective_stride={self.cnn_effective_stride})."
            )
        h, w_ = H // p, W // p
        x = imgs.reshape(B, c, h, p, w_, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h * w_, p * p * c)
        return x

    def unpatchify(
        self, x: torch.Tensor, grid_h: int, grid_w: int,
    ) -> torch.Tensor:
        """
        Reconstruct images from patch pixel values.

        Args:
            x:       ``[B, grid_h * grid_w, pixel_patch_size**2 * C]``
            grid_h:  number of patch rows (in CNN feature / token space)
            grid_w:  number of patch columns
        Returns:
            ``[B, C, grid_h * pixel_patch_size, grid_w * pixel_patch_size]``
        """
        p = self.pixel_patch_size
        c = self.in_channels
        B = x.shape[0]
        x = x.reshape(B, grid_h, grid_w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
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
        B, N, D = x.shape
        noise = torch.rand(B, N, device=x.device)
        if pad_mask is not None:
            noise = noise.masked_fill(pad_mask, float("inf"))
            n_content = (~pad_mask).float().sum(dim=1)
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
        ar = torch.arange(grid_h * grid_w, device=device)
        return (ar // grid_w).float(), (ar % grid_w).float()

    def _encoder_token_positions(
        self,
        ids_keep: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        patch_h, patch_w = self._patch_grid_positions(grid_h, grid_w, device)
        z = torch.zeros(B, 1, device=device, dtype=patch_h.dtype)
        ph = torch.cat([z, patch_h.unsqueeze(0).expand(B, -1)], dim=1)
        pw = torch.cat([z, patch_w.unsqueeze(0).expand(B, -1)], dim=1)
        return ph, pw

    # -----------------------------------------------------------------
    # Internal: CNN preprocessing + pad_mask downsampling
    # -----------------------------------------------------------------

    def _apply_cnn(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run CNN on ``x`` and downsample ``pad_mask`` to match CNN output size."""
        if self.cnn is None:
            return x, pad_mask
        x = self.cnn(x)
        if pad_mask is not None and self.cnn_effective_stride > 1:
            S = self.cnn_effective_stride
            # A CNN cell is padding if ANY of its S×S source pixels is padding.
            pad_mask = F.max_pool2d(
                pad_mask.float(), kernel_size=S, stride=S, padding=0
            ).bool()
        return x, pad_mask

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
            x:        ``[B, C, H, W]``
            pad_mask: ``[B, 1, H, W]`` boolean (True = pad pixel), or ``None``.

        Returns:
            latent:      ``[B, N_visible + 1, embed_dim]``
            mask:        ``[B, N]``
            ids_restore: ``[B, N]``
            grid_h:      token rows
            grid_w:      token columns
        """
        x, cnn_pad_mask = self._apply_cnn(x, pad_mask)

        patch_pad_mask: Optional[torch.Tensor] = None
        if cnn_pad_mask is not None:
            patch_pad_mask = _pixel_pad_mask_to_patch(cnn_pad_mask, self.patch_size)

        x, grid_h, grid_w = self.patch_embed(x)

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

        Returns:
            pred: ``[B, N, pixel_patch_size**2 * in_channels]``
        """
        x = self.decoder_embed(x)

        N = grid_h * grid_w
        num_masked = N - (x.shape[1] - 1)
        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1)

        x_no_cls = x[:, 1:, :]
        x_ = torch.cat([x_no_cls, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]),
        )

        x = torch.cat([x[:, :1, :], x_], dim=1)

        B = x.shape[0]
        pos_h, pos_w = self._full_token_positions(B, grid_h, grid_w, x.device)

        for blk in self.decoder_blocks:
            x = blk(x, pos_h, pos_w)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x[:, 1:, :])
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
        Weighted L1 + L2 reconstruction loss on masked content patches.

        Patchification uses ``pixel_patch_size`` so each patch corresponds to
        the full original-pixel receptive field of one token.

        Args:
            imgs:     ``[B, C, H, W]`` (original input before CNN).
            pred:     ``[B, N, pixel_patch_size**2 * C]``
            mask:     ``[B, N]`` (1 = masked)
            pad_mask: ``[B, 1, H, W]`` boolean pixel mask, or ``None``.
        """
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        diff = pred - target
        l1 = diff.abs().mean(dim=-1)
        l2 = (diff ** 2).mean(dim=-1)
        loss = self.l1_loss_weight * l1 + self.l2_loss_weight * l2

        if self.fft_loss_weight > 0:
            p = self.pixel_patch_size
            B, N, _ = pred.shape
            pred_s = pred.reshape(B * N, self.in_channels, p, p)
            tgt_s  = target.reshape(B * N, self.in_channels, p, p)
            fft_diff = (
                torch.fft.rfft2(pred_s, norm="ortho")
                - torch.fft.rfft2(tgt_s, norm="ortho")
            )
            fft_loss = fft_diff.abs().mean(dim=(-3, -2, -1)).reshape(B, N)
            loss = loss + self.fft_loss_weight * fft_loss

        effective_mask = mask
        if pad_mask is not None:
            # pad_mask is at original pixel resolution; use pixel_patch_size
            patch_pad = _pixel_pad_mask_to_patch(pad_mask, self.pixel_patch_size)
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
        (no masking).  The CNN feature extractor runs first if configured.

        Args:
            imgs:     ``[B, C, H, W]``
            pad_mask: ``[B, 1, H, W]`` boolean (True = pad), or ``None``.
            pool:     ``"mean"`` *(default)* or ``"cls"``.

        Returns:
            ``[B, embed_dim]``
        """
        if pool not in ("mean", "cls"):
            raise ValueError(f"pool must be 'mean' or 'cls', got {pool!r}")

        x, cnn_pad_mask = self._apply_cnn(imgs, pad_mask)

        x, grid_h, grid_w = self.patch_embed(x)

        patch_pad: Optional[torch.Tensor] = None
        if cnn_pad_mask is not None:
            patch_pad = _pixel_pad_mask_to_patch(cnn_pad_mask, self.patch_size)
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

        patches = x[:, 1:]
        if patch_pad is not None:
            keep = (~patch_pad).unsqueeze(-1).float()
            return (patches * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        return patches.mean(dim=1)

    def get_encoder(self) -> "CNNMaskedAutoencoderViT":
        return self


# =====================================================================
# Factory
# =====================================================================

def create_cnn_mae_vit(
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
    cnn_layer_channels: Optional[List[int]] = None,
    cnn_kernel_sizes: Union[int, List[int]] = 3,
    cnn_strides: Union[int, List[int]] = 1,
) -> CNNMaskedAutoencoderViT:
    """
    Create a CNN + MAE ViT for grayscale ultrasound images.

    Default CNN: two layers, channels [32, 64], kernel 3×3, stride 1
    (effective_stride=1, pixel_patch_size=patch_size — same granularity as
    plain ViT but with learned local features before tokenisation).

    Set ``cnn_strides=[1, 2]`` to halve the spatial resolution before patching
    (pixel_patch_size doubles, so images must be padded to multiples of
    ``patch_size * 2``).

    ``max_image_height`` is stored as metadata so the preprocessing pipeline
    can be replayed from a checkpoint.
    """
    if cnn_layer_channels is None:
        cnn_layer_channels = [32, 64]
    return CNNMaskedAutoencoderViT(
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
        cnn_layer_channels=cnn_layer_channels,
        cnn_kernel_sizes=cnn_kernel_sizes,
        cnn_strides=cnn_strides,
    )
