"""
Lightweight U-Net for binary ultrasound conus segmentation.

Encoder: pretrained timm CNN backbone (features_only=True, in_chans=1).
Decoder: 4 up-blocks with skip connections from encoder feature maps.
Head:    1×1 Conv → single-channel logit map.

Output spatial size matches input spatial size.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _UNetDecoder(nn.Module):
    def __init__(self, enc_channels: List[int], dec_channels: List[int]):
        super().__init__()
        # enc_channels: finest → coarsest  (e.g. [16, 24, 40, 112, 320])
        # dec_channels: width at each up-step (len = len(enc_channels)-1)
        assert len(dec_channels) == len(enc_channels) - 1
        self.blocks = nn.ModuleList()
        in_ch = enc_channels[-1]
        for i, dec_ch in enumerate(dec_channels):
            skip_ch = enc_channels[-(i + 2)]
            self.blocks.append(_conv_block(in_ch + skip_ch, dec_ch))
            in_ch = dec_ch
        self.out_channels = in_ch

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        for i, block in enumerate(self.blocks):
            skip = features[-(i + 2)]
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return x


class ConusUNet(nn.Module):
    """
    U-Net conus segmentation model.

    Args:
        backbone:         timm model name, must support ``features_only=True``.
                          Default: ``"efficientnet_b0"`` (~5 M params).
        decoder_channels: Feature widths in the 4 decoder up-stages.
        pretrained:       Download pretrained backbone weights.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        decoder_channels: List[int] = (256, 128, 64, 32),
        pretrained: bool = True,
    ):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required: pip install timm")

        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            in_chans=1,
        )
        enc_ch = self.encoder.feature_info.channels()   # e.g. [16, 24, 40, 112, 320]

        self.decoder = _UNetDecoder(enc_ch, list(decoder_channels))
        self.head    = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)

        self.backbone         = backbone
        self.decoder_channels = list(decoder_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits ``[B, 1, H, W]`` at the same spatial size as input."""
        features = self.encoder(x)
        decoded  = self.decoder(features)
        logits   = self.head(decoded)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns binary mask ``[B, 1, H, W]``."""
        return (torch.sigmoid(self.forward(x)) > threshold).float()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
