"""
Custom Vision Transformer with MAE pre-training for ultrasound embeddings.

Quick start::

    from embeddings.vit import create_mae_vit, train_mae, load_checkpoint

    # Create a model
    model = create_mae_vit()

    # Train on a dataset
    model, history = train_mae(dataset=my_dataset, epochs=100)

    # Load from checkpoint
    model, info = load_checkpoint("checkpoints/mae_final.pt")

    # Extract embeddings
    embeddings = model.encode(images)  # [B, embed_dim]
"""

from embeddings.vit.model import (
    MaskedAutoencoderViT,
    create_mae_vit,
)
from embeddings.vit.train import (
    load_checkpoint,
    train_mae,
    visualize_reconstruction,
)

__all__ = [
    "MaskedAutoencoderViT",
    "create_mae_vit",
    "train_mae",
    "load_checkpoint",
    "visualize_reconstruction",
]

