"""
Re-exports from baselines.mae_classify for backward compatibility.

The linear probe implementation has moved to baselines/mae_classify/.
"""

from baselines.mae_classify.model import (
    ClassificationDatasetWrapper as _ClassificationDatasetWrapper,
    LinearProbe,
    default_image_transform as _default_image_transform,
    pad_classify_collate,
)
from baselines.mae_classify.train import (
    gather_predictions as linear_probe_gather_predictions,
    confusion_matrix as linear_probe_confusion_matrix,
    load_linear_probe_from_checkpoint,
    make_fetal_planes_probe_dataloader,
    plot_linear_probe_history,
    train_linear_probe,
)

__all__ = [
    "LinearProbe",
    "pad_classify_collate",
    "train_linear_probe",
    "load_linear_probe_from_checkpoint",
    "make_fetal_planes_probe_dataloader",
    "linear_probe_gather_predictions",
    "linear_probe_confusion_matrix",
    "plot_linear_probe_history",
]
