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
from typing import Optional, Tuple

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
                           unit-variance before computing MSE loss.
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
    ):
        super().__init__()

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
        # Predict pixel values for each patch
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_channels, bias=True
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
        self, x: torch.Tensor, mask_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Per-sample random masking by shuffling patch indices.

        Args:
            x: [B, N, D]  (patch embeddings, no CLS)
            mask_ratio: fraction of patches to mask

        Returns:
            x_masked: [B, N_visible, D]
            mask:     [B, N]  binary, 1 = masked (removed), 0 = kept
            ids_restore: [B, N]  indices to unshuffle
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)       # [B, N]
        ids_shuffle = noise.argsort(dim=1)                # ascend: small kept
        ids_restore = ids_shuffle.argsort(dim=1)

        # Keep first num_keep tokens
        ids_keep = ids_shuffle[:, :num_keep]              # [B, num_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # Binary mask: 0 = kept, 1 = removed
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    # -----------------------------------------------------------------
    # Encoder
    # -----------------------------------------------------------------

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode only the *visible* patches (training path).

        Args:
            x: [B, C, H, W]

        Returns:
            latent:      [B, N_visible + 1, embed_dim]  (includes CLS)
            mask:        [B, N]
            ids_restore: [B, N]
        """
        # Patch embed
        x = self.patch_embed(x)                          # [B, N, D]

        # Add positional embed (skip CLS slot at index 0)
        x = x + self.pos_embed[:, 1:, :]

        # Mask
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Prepend CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)            # [B, 1+N_vis, D]

        # Transformer blocks
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
            pred: [B, N, patch_size**2 * in_channels]
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

        # Predict pixel values (skip CLS)
        x = self.decoder_pred(x[:, 1:, :])               # [B, N, p*p*C]

        return x

    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------

    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE loss on masked patches only.

        Args:
            imgs: [B, C, H, W]
            pred: [B, N, p*p*C]
            mask: [B, N]  (1 = masked)

        Returns:
            Scalar loss.
        """
        target = self.patchify(imgs)                      # [B, N, p*p*C]

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                          # [B, N]

        # Average over masked patches only
        loss = (loss * mask).sum() / mask.sum()
        return loss

    # -----------------------------------------------------------------
    # Full forward (training)
    # -----------------------------------------------------------------

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full MAE forward pass (training).

        Args:
            imgs: [B, C, H, W]

        Returns:
            loss:  scalar reconstruction loss
            pred:  [B, N, p*p*C]
            mask:  [B, N]
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    # -----------------------------------------------------------------
    # Embedding extraction (inference)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS-token embeddings by feeding *all* patches through
        the encoder (no masking).

        Args:
            imgs: [B, C, H, W]

        Returns:
            [B, embed_dim]
        """
        x = self.patch_embed(imgs)                       # [B, N, D]
        x = x + self.pos_embed[:, 1:, :]

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
    )

