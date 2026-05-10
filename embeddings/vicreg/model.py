"""
VICReg (Variance-Invariance-Covariance Regularization) for grayscale ultrasound ViT.

Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for
Self-Supervised Learning", ICLR 2022.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from embeddings.vit.model import MaskedAutoencoderViT, create_mae_vit


def _build_projector(in_dim: int, proj_dim: int, n_layers: int) -> nn.Sequential:
    """3-layer MLP expander: Linear → BN → ReLU → ... → Linear (no BN on last layer)."""
    layers: list = []
    dim = in_dim
    for i in range(n_layers):
        out = proj_dim
        layers.append(nn.Linear(dim, out, bias=False))
        if i < n_layers - 1:
            layers.append(nn.BatchNorm1d(out))
            layers.append(nn.ReLU(inplace=True))
        dim = out
    return nn.Sequential(*layers)


class VICRegModel(nn.Module):
    """
    VICReg: ViT encoder + MLP projector head.

    The encoder is a ``MaskedAutoencoderViT`` (only encoder weights are trained;
    decoder is carried for checkpoint compatibility with MAE warm-starts).
    ``encode()`` returns encoder representations for downstream use / RankME.
    ``forward()`` returns projected representations (z1, z2) for loss computation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projector_dim: int = 2048,
        projector_layers: int = 3,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = _build_projector(encoder.embed_dim, projector_dim, projector_layers)
        self.projector_dim = projector_dim
        self.projector_layers = projector_layers

    # ------------------------------------------------------------------

    def encode(
        self,
        imgs: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[B, embed_dim] encoder representations (no projector). Used for RankME / probing."""
        return self.encoder.encode(imgs, pad_mask=pad_mask)

    def forward(
        self,
        imgs1: torch.Tensor,
        pad_mask1: Optional[torch.Tensor],
        imgs2: torch.Tensor,
        pad_mask2: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns projected representations (z1, z2), each [B, projector_dim]."""
        z1 = self.projector(self.encode(imgs1, pad_mask=pad_mask1))
        z2 = self.projector(self.encode(imgs2, pad_mask=pad_mask2))
        return z1, z2

    # ------------------------------------------------------------------

    def vicreg_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
    ) -> Tuple[torch.Tensor, float, float, float]:
        """
        VICReg loss.

        Returns:
            (total_loss, sim_scalar, var_scalar, cov_scalar)
            where scalar values are detached floats for logging.
        """
        N, D = z1.shape

        # Invariance: penalize distance between two views of the same image.
        sim = F.mse_loss(z1, z2)

        # Variance: keep per-dimension std >= 1 via a hinge.
        std1 = z1.var(dim=0).add_(1e-4).sqrt()
        std2 = z2.var(dim=0).add_(1e-4).sqrt()
        var = (F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean()) / 2.0

        # Covariance: decorrelate dimensions by penalizing off-diagonal entries.
        z1c = z1 - z1.mean(dim=0)
        z2c = z2 - z2.mean(dim=0)
        cov1 = (z1c.T @ z1c) / max(N - 1, 1)
        cov2 = (z2c.T @ z2c) / max(N - 1, 1)
        off1 = cov1.pow(2).sum() - cov1.diagonal().pow(2).sum()
        off2 = cov2.pow(2).sum() - cov2.diagonal().pow(2).sum()
        cov = (off1 + off2) / D

        loss = sim_coeff * sim + std_coeff * var + cov_coeff * cov
        return loss, sim.item(), var.item(), cov.item()

    # ------------------------------------------------------------------

    def encoder_parameters(self):
        """Yields only encoder parameters — excludes MAE decoder/mask_token and projector."""
        for n, p in self.encoder.named_parameters():
            if not n.startswith(('decoder', 'mask_token')):
                yield p

    def trainable_parameters(self):
        """Encoder parameters + projector parameters (used for optimizer)."""
        yield from self.encoder_parameters()
        yield from self.projector.parameters()


# =====================================================================
# Factory
# =====================================================================

def create_vicreg(
    patch_size: int = 16,
    embed_dim: int = 512,
    depth: int = 8,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    projector_dim: int = 2048,
    projector_layers: int = 3,
    max_image_height: Optional[int] = None,
) -> VICRegModel:
    """
    Create a VICReg model with a ViT encoder and MLP projector.

    Encoder matches the ultrasound MAE ViT (grayscale, 2D RoPE, variable-size images).
    A stub decoder is included so MAE checkpoints can be loaded for encoder warm-starting
    via ``mae_init_checkpoint`` in ``train_vicreg``; it is not trained during VICReg.
    """
    encoder = create_mae_vit(
        patch_size=patch_size,
        in_channels=1,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_embed_dim=128,
        decoder_depth=1,
        decoder_num_heads=1,
        mlp_ratio=mlp_ratio,
        norm_pix_loss=False,
        clip_pixel_pred=False,
        max_image_height=max_image_height,
    )
    return VICRegModel(encoder, projector_dim=projector_dim, projector_layers=projector_layers)


def create_vicreg_timm(
    timm_model: str,
    projector_dim: int = 2048,
    projector_layers: int = 3,
    pretrained: bool = True,
    channel_avg: bool = True,
) -> VICRegModel:
    """
    Create a VICReg model with a pretrained timm ViT as the encoder.

    The timm ViT is wrapped by ``TimmViTEncoder`` which adapts it to 1-channel
    grayscale input and exposes the same ``encode()`` interface.  Variable-size
    inputs are supported via positional-embedding interpolation (dynamic_img_size).

    Example::

        model = create_vicreg_timm("vit_small_patch16_224.dino")
    """
    from embeddings.timm_encoder import TimmViTEncoder
    encoder = TimmViTEncoder(timm_model, pretrained=pretrained, channel_avg=channel_avg)
    return VICRegModel(encoder, projector_dim=projector_dim, projector_layers=projector_layers)
