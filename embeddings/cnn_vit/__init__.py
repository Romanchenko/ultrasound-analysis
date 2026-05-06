"""
CNN + Vision Transformer with MAE pre-training for ultrasound embeddings.

A configurable multi-layer CNN extracts feature maps from raw images before
ViT tokenisation.  All other architecture, training, and inference options
mirror ``embeddings.vit``.

Quick start::

    from embeddings.cnn_vit import create_cnn_mae_vit, train_mae, load_checkpoint

    # Create a model (2-layer CNN → 16×16 ViT patches over feature maps)
    model = create_cnn_mae_vit(
        cnn_layer_channels=[32, 64],
        cnn_kernel_sizes=3,
        cnn_strides=1,           # stride 1 → pixel_patch_size = patch_size
    )

    # With spatial downsampling (effective stride 2 → pixel_patch_size = 32)
    model = create_cnn_mae_vit(
        cnn_layer_channels=[32, 64],
        cnn_kernel_sizes=[3, 3],
        cnn_strides=[1, 2],
        patch_size=16,
        max_image_height=224,    # must be divisible by 16*2 = 32
    )

    # Train on a dataset
    model, history = train_mae(
        dataset=my_dataset,
        cnn_layer_channels=[32, 64],
        cnn_strides=[1, 2],
        max_image_height=224,
        epochs=200,
    )

    # Load from checkpoint
    model, info = load_checkpoint("checkpoints/mae_final.pt")

    # Extract embeddings
    embeddings = model.encode(images)  # [B, embed_dim]
"""

from embeddings.cnn_vit.model import (
    CNNFeatureExtractor,
    CNNMaskedAutoencoderViT,
    create_cnn_mae_vit,
)
from embeddings.cnn_vit.train import (
    dump_mae_training_metrics_artifacts,
    load_checkpoint,
    train_mae,
    visualize_reconstruction,
)

__all__ = [
    "CNNFeatureExtractor",
    "CNNMaskedAutoencoderViT",
    "create_cnn_mae_vit",
    "train_mae",
    "load_checkpoint",
    "visualize_reconstruction",
    "dump_mae_training_metrics_artifacts",
]
