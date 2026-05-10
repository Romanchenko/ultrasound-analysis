"""
RankME: effective-rank metric for self-supervised embeddings.
Garrido et al., "RankMe: Assessing the downstream performance of
pretrained self-supervised representations by their rank", ICML 2023.
"""
import torch
from torch.utils.data import DataLoader


def compute_rank_me(embeddings: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Args:
        embeddings: [N, D] unnormalized embeddings.
    Returns:
        Effective rank = exp(Shannon entropy of normalized singular values).
        Higher = embeddings span more of the feature space (less collapsed).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"expected [N, D] tensor, got shape {tuple(embeddings.shape)}")
    norms = embeddings.norm(dim=1, keepdim=True).clamp_min(eps)
    z = embeddings / norms
    # svdvals is cheaper than full SVD since we only need singular values.
    # Cast to float32 in case embeddings come from an AMP (fp16) model.
    s = torch.linalg.svdvals(z.float().cpu())
    s = s.clamp_min(0.0)
    p = s / s.sum().clamp_min(eps)
    entropy = -(p * p.clamp_min(eps).log()).sum().item()
    return float(torch.exp(torch.tensor(entropy)).item())


def collect_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Collect [N, D] embeddings for all batches in a dataloader.

    Expects ``(images, pad_masks)`` batches (the format produced by
    ``mae_pad_collate`` in both ``embeddings.vit`` and ``embeddings.cnn_vit``).
    Calls ``model.encode(images, pad_mask=pad_masks)`` → [B, embed_dim].

    Args:
        model:      Any model with an ``encode(imgs, pad_mask=...)`` method.
        dataloader: Yields ``(images, pad_masks)`` tuples.
        device:     Device to run inference on.

    Returns:
        ``[N, embed_dim]`` tensor on CPU.
    """
    model.eval()
    chunks = []
    with torch.no_grad():
        for images, pad_masks in dataloader:
            images    = images.to(device, non_blocking=True)
            pad_masks = pad_masks.to(device, non_blocking=True)
            chunks.append(model.encode(images, pad_mask=pad_masks).cpu())
    return torch.cat(chunks, dim=0)
