"""
Dataset that translates images from domain A to domain B on the fly
using a trained CycleGAN generator.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Callable

from domain_adaptation.cyclegan.model import CycleGAN


class CycleGANTranslatedDataset(Dataset):
    """
    Wraps a source (domain A) dataset and a CycleGAN model to produce
    translated (domain B) images on the fly.

    Each __getitem__ call:
      1. Fetches an image from the source dataset.
      2. Passes it through the G_A2B generator.
      3. Returns the translated image (and optionally the original).

    The generator runs in eval mode with torch.no_grad() so no gradients
    are tracked, keeping memory usage low.

    Usage:
        >>> from domain_adaptation.cyclegan.model import create_cyclegan_model
        >>> model = create_cyclegan_model(input_channels_a=1, input_channels_b=1)
        >>> model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
        >>> translated = CycleGANTranslatedDataset(model, source_dataset, device='cpu')
        >>> sample = translated[0]
        >>> sample['image']        # translated image tensor
        >>> sample['source_image'] # original image tensor (if return_source=True)
    """

    def __init__(
        self,
        cyclegan_model: CycleGAN,
        source_dataset: Dataset,
        device: Optional[torch.device] = None,
        return_source: bool = False,
        post_transform: Optional[Callable] = None,
        b2a: bool = False,
    ):
        """
        Args:
            cyclegan_model: Trained CycleGAN model. Only G_A2B is used.
            source_dataset: Domain-A dataset (any torch Dataset that returns
                            a dict with 'image' key or a plain tensor).
            device: Device on which to run the generator. If None, auto-detected.
            return_source: If True, include the original (source) image in the
                           returned dict under 'source_image'.
            post_transform: Optional callable applied to the translated image
                            tensor *after* generation (e.g. denormalization,
                            clipping, dtype conversion).
        """
        self.generator = cyclegan_model.G_A2B if not b2a else cyclegan_model.G_B2A
        self.generator.eval()

        self.source_dataset = source_dataset
        self.return_source = return_source
        self.post_transform = post_transform

        if device is None:
            self.device = next(self.generator.parameters()).device
        else:
            self.device = torch.device(device)
            self.generator = self.generator.to(self.device)

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, idx: int) -> dict:
        """
        Fetch a source image, translate it A→B, and return the result.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with at least:
                'image'  – translated (domain-B) image tensor [C, H, W]
            Optionally:
                'source_image' – original (domain-A) image tensor [C, H, W]
            Any extra keys from the source sample (labels, metadata, etc.)
            are forwarded unchanged.
        """
        sample = self.source_dataset[idx]

        # --- extract image tensor from sample ---
        extra = {}
        if isinstance(sample, dict):
            source_image = sample['image']
            # Keep all non-image keys so labels / metadata pass through
            extra = {k: v for k, v in sample.items() if k != 'image'}
        else:
            source_image = sample

        if not isinstance(source_image, torch.Tensor):
            raise TypeError(
                f"Expected image to be a torch.Tensor, got {type(source_image)}. "
                "Make sure the source dataset returns tensors."
            )

        # --- translate A → B ---
        # Add batch dimension: [C, H, W] -> [1, C, H, W]
        input_tensor = source_image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            translated = self.generator(input_tensor)

        # Remove batch dimension: [1, C, H, W] -> [C, H, W]
        translated = translated.squeeze(0).cpu()

        if self.post_transform is not None:
            translated = self.post_transform(translated)

        # --- build result ---
        result = {**extra, 'image': translated}

        if self.return_source:
            result['source_image'] = source_image

        return result

