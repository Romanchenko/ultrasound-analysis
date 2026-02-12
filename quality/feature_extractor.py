"""
Shared feature extraction module for quality metrics (FID, MMD, etc.).

Supports multiple pretrained backbones:
    - 'resnet50'     : torchvision ResNet-50 with ImageNet-1K v2 weights
    - 'radimagenet'  : ResNet-50 with RadImageNet weights loaded from a
                       Keras H5 file (converted to PyTorch on the fly)
    - nn.Module      : any custom model passed directly

RadImageNet weights are expected as a Keras H5 file (.h5).  On first use
the weights are converted to a PyTorch state dict and cached as a .pt file
next to the original H5 so the conversion only runs once.
"""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


# =====================================================================
# Public API
# =====================================================================

def get_features_model(
    model_name: Union[str, nn.Module] = 'resnet50',
    device: Optional[torch.device] = None,
    weights_path: Optional[str] = None,
) -> nn.Module:
    """
    Build a feature-extraction model (classification head removed).

    Args:
        model_name:
            * ``'resnet50'``     – ImageNet pretrained ResNet-50.
            * ``'radimagenet'``  – ResNet-50 with RadImageNet weights.
              Requires *weights_path* pointing to a Keras ``.h5`` file
              (or a previously converted ``.pt`` file).
            * An ``nn.Module``   – used as-is (caller is responsible for
              removing the classification head).
        device:       Target device.  Auto-detected when *None*.
        weights_path: Path to weight file.  Required for ``'radimagenet'``.

    Returns:
        ``nn.Module`` in eval mode on the requested device.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(model_name, nn.Module):
        model = model_name
        model.eval()
        return model.to(device)

    name = model_name.lower()

    if name == 'resnet50':
        return _get_resnet50_imagenet(device)

    if name == 'radimagenet':
        if weights_path is None:
            raise ValueError(
                "weights_path is required for RadImageNet. "
                "Provide the path to the Keras .h5 file (or a converted .pt file)."
            )
        return _get_radimagenet(weights_path, device)

    raise ValueError(
        f"Unknown model_name: '{model_name}'. "
        f"Supported: 'resnet50', 'radimagenet', or pass an nn.Module directly."
    )


# =====================================================================
# ImageNet ResNet-50
# =====================================================================

def _get_resnet50_imagenet(device: torch.device) -> nn.Module:
    """Load torchvision ResNet-50 (ImageNet-1K v2), remove FC head."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = nn.Sequential(*list(model.children())[:-1])  # drop avgpool+fc → keep up to avgpool
    model.eval()
    return model.to(device)


# =====================================================================
# RadImageNet (Keras H5 → PyTorch)
# =====================================================================

def _get_radimagenet(weights_path: str, device: torch.device) -> nn.Module:
    """
    Load ResNet-50 with RadImageNet weights.

    If *weights_path* ends with ``.pt``, it is loaded directly as a
    PyTorch state dict.  If it ends with ``.h5``, it is converted from
    Keras format first (and the result is cached as ``<stem>_pytorch.pt``
    next to the original file).
    """
    weights_path = Path(weights_path)

    if weights_path.suffix == '.pt':
        state_dict = torch.load(weights_path, map_location='cpu')
    elif weights_path.suffix in ('.h5', '.hdf5'):
        # Check for cached .pt
        cached_pt = weights_path.with_name(weights_path.stem + '_pytorch.pt')
        if cached_pt.exists():
            print(f"Loading cached PyTorch weights from {cached_pt}")
            state_dict = torch.load(cached_pt, map_location='cpu')
        else:
            print(f"Converting Keras H5 weights to PyTorch: {weights_path}")
            state_dict = _convert_keras_h5_to_pytorch_state_dict(weights_path)
            torch.save(state_dict, cached_pt)
            print(f"Cached converted weights to {cached_pt}")
    else:
        raise ValueError(
            f"Unsupported weight file format: '{weights_path.suffix}'. "
            f"Expected .h5, .hdf5, or .pt"
        )

    # Build a blank ResNet-50 and load the converted weights.
    # We strip the FC head *before* loading so that weights files without
    # a prediction layer (e.g. Keras ``_notop`` models) work out of the box.
    model = resnet50(weights=None)

    has_fc = 'fc.weight' in state_dict
    if has_fc:
        # Match the FC size if it differs from the default 1000 classes
        fc_weight = state_dict['fc.weight']
        if fc_weight.shape[0] != 1000:
            model.fc = nn.Linear(model.fc.in_features, fc_weight.shape[0])
    else:
        # No FC in the weight file — remove it from the model so
        # load_state_dict doesn't complain about missing keys.
        model.fc = nn.Identity()

    model.load_state_dict(state_dict, strict=True)

    # Remove classification head (FC / Identity) → keep up to avgpool
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model.to(device)


def convert_keras_h5_to_pytorch(h5_path: str, output_path: Optional[str] = None) -> str:
    """
    Standalone utility: convert a Keras ResNet-50 H5 file to a PyTorch
    ``.pt`` state dict.

    Args:
        h5_path:     Path to the Keras ``.h5`` file.
        output_path: Where to save the ``.pt`` file.
                     Defaults to ``<h5_stem>_pytorch.pt`` in the same dir.

    Returns:
        Path to the saved ``.pt`` file.
    """
    h5_path = Path(h5_path)
    if output_path is None:
        output_path = h5_path.with_name(h5_path.stem + '_pytorch.pt')
    else:
        output_path = Path(output_path)

    state_dict = _convert_keras_h5_to_pytorch_state_dict(h5_path)
    torch.save(state_dict, output_path)
    print(f"Saved PyTorch state dict to {output_path}")
    return str(output_path)


# =====================================================================
# Keras H5 → PyTorch weight conversion internals
# =====================================================================

# ResNet-50 stage → PyTorch layer name, and block counts per stage
_STAGE_MAP = {
    'conv2': ('layer1', 3),
    'conv3': ('layer2', 4),
    'conv4': ('layer3', 6),
    'conv5': ('layer4', 3),
}


def _convert_keras_h5_to_pytorch_state_dict(h5_path: Path) -> dict:
    """
    Read a Keras ResNet-50 H5 file and return a PyTorch ``state_dict``.

    Handles both full-model saves (``model.save()``) and weight-only
    saves (``model.save_weights()``).
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required to convert Keras H5 weights. "
            "Install with: pip install h5py"
        )

    f = h5py.File(str(h5_path), 'r')

    # Determine root group: full model saves nest under 'model_weights'
    if 'model_weights' in f:
        root = f['model_weights']
    else:
        root = f

    state_dict = {}

    # --- initial conv + bn ---
    _convert_conv(root, 'conv1_conv', state_dict, 'conv1')
    _convert_bn(root, 'conv1_bn', state_dict, 'bn1')

    # --- residual stages ---
    for keras_stage, (pt_layer, num_blocks) in _STAGE_MAP.items():
        for block_idx in range(1, num_blocks + 1):
            pt_block = f'{pt_layer}.{block_idx - 1}'
            keras_prefix = f'{keras_stage}_block{block_idx}'

            # Three convolutions per bottleneck block: sub 1, 2, 3
            for sub in (1, 2, 3):
                _convert_conv(root, f'{keras_prefix}_{sub}_conv',
                              state_dict, f'{pt_block}.conv{sub}')
                _convert_bn(root, f'{keras_prefix}_{sub}_bn',
                            state_dict, f'{pt_block}.bn{sub}')

            # Shortcut / downsample (sub 0) — only present in some blocks
            shortcut_conv_name = f'{keras_prefix}_0_conv'
            if _layer_exists(root, shortcut_conv_name):
                _convert_conv(root, shortcut_conv_name,
                              state_dict, f'{pt_block}.downsample.0')
                _convert_bn(root, f'{keras_prefix}_0_bn',
                            state_dict, f'{pt_block}.downsample.1')

    # --- FC (predictions) ---
    if _layer_exists(root, 'predictions'):
        _convert_dense(root, 'predictions', state_dict, 'fc')

    f.close()

    # Sanity check: make sure we got a reasonable number of keys
    if len(state_dict) < 100:
        raise RuntimeError(
            f"Only extracted {len(state_dict)} parameters from H5 file. "
            f"Expected ~160+ for ResNet-50. The file may not be a "
            f"standard Keras ResNet-50."
        )

    print(f"Converted {len(state_dict)} parameter tensors from Keras H5")
    return state_dict


# ---------- helpers for reading individual layers ----------

def _layer_exists(root, layer_name: str) -> bool:
    """Check whether a Keras layer group exists in the H5 root."""
    return layer_name in root


def _get_weight(root, layer_name: str, weight_name: str) -> np.ndarray:
    """
    Read a single weight array from the H5 file.

    Keras nests weights as ``root/<layer_name>/<layer_name>/<weight_name>``.
    """
    group = root[layer_name]
    # Navigate the nested group (Keras convention)
    if layer_name in group:
        group = group[layer_name]
    return np.array(group[weight_name])


def _convert_conv(root, keras_name: str, state_dict: dict, pt_name: str):
    """Convert a Keras Conv2D layer → PyTorch Conv2d weight."""
    kernel = _get_weight(root, keras_name, 'kernel:0')
    # Keras: [H, W, C_in, C_out] → PyTorch: [C_out, C_in, H, W]
    kernel = np.transpose(kernel, (3, 2, 0, 1))
    state_dict[f'{pt_name}.weight'] = torch.from_numpy(kernel.copy())


def _convert_bn(root, keras_name: str, state_dict: dict, pt_name: str):
    """Convert a Keras BatchNormalization layer → PyTorch BatchNorm2d."""
    gamma = _get_weight(root, keras_name, 'gamma:0')
    beta = _get_weight(root, keras_name, 'beta:0')
    moving_mean = _get_weight(root, keras_name, 'moving_mean:0')
    moving_var = _get_weight(root, keras_name, 'moving_variance:0')

    state_dict[f'{pt_name}.weight'] = torch.from_numpy(gamma.copy())
    state_dict[f'{pt_name}.bias'] = torch.from_numpy(beta.copy())
    state_dict[f'{pt_name}.running_mean'] = torch.from_numpy(moving_mean.copy())
    state_dict[f'{pt_name}.running_var'] = torch.from_numpy(moving_var.copy())
    state_dict[f'{pt_name}.num_batches_tracked'] = torch.tensor(0, dtype=torch.long)


def _convert_dense(root, keras_name: str, state_dict: dict, pt_name: str):
    """Convert a Keras Dense layer → PyTorch Linear weight + bias."""
    kernel = _get_weight(root, keras_name, 'kernel:0')
    bias = _get_weight(root, keras_name, 'bias:0')
    # Keras: [in_features, out_features] → PyTorch: [out_features, in_features]
    kernel = np.transpose(kernel, (1, 0))
    state_dict[f'{pt_name}.weight'] = torch.from_numpy(kernel.copy())
    state_dict[f'{pt_name}.bias'] = torch.from_numpy(bias.copy())

