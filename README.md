# Ultrasound Analysis with DinoV2

Fine-tuning DinoV2 model for self-supervised learning on ultrasound images.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

- `dataloader.py`: Module for loading ultrasound images from folders
- `fine_tune_dinov2.py`: Module for fine-tuning DinoV2 with self-supervised learning
- `train.py`: Main training script
- `visualize_data.py`: Module for visualizing and inspecting dataloader data

## Usage

### Training with DDTI-style Directory Structure

The dataloader supports DDTI-style directory structure (based on [TRFE-Net](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation)):

```
data/
  train/
    image/
      *.jpg, *.png, etc.
  val/
    image/
      *.jpg, *.png, etc.
```

```bash
python train.py \
    --train_root /path/to/data/train \
    --val_root /path/to/data/val \
    --use_ddti \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --checkpoint_dir ./checkpoints
```

### Training with Folder-based Structure (Legacy)

```bash
python train.py \
    --train_folders /path/to/train/images /path/to/more/train/images \
    --val_folders /path/to/val/images \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --checkpoint_dir ./checkpoints
```

### Key Arguments

**DDTI Mode:**
- `--train_root`: Root directory containing `image/` subdirectory with training images
- `--val_root`: (Optional) Root directory containing `image/` subdirectory with validation images
- `--use_ddti`: Flag to use DDTI-style directory structure

**Legacy Mode:**
- `--train_folders`: One or more folders containing training images
- `--val_folders`: (Optional) Folders containing validation images

**Common Arguments:**
- `--model_name`: DinoV2 model name (default: `facebook/dinov2-small`)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--freeze_backbone`: Freeze the DinoV2 backbone during training
- `--checkpoint_dir`: Directory to save checkpoints (default: `./checkpoints`)

### Training in Jupyter Notebook

The training script can also be used directly in Jupyter notebook cells:

```python
from train import train_model

# Train with DDTI structure
train_model(
    train_root='./data/train',
    val_root='./data/val',
    use_ddti=True,
    batch_size=32,
    epochs=50,
    learning_rate=1e-4,
    checkpoint_dir='./checkpoints'
)

# Or with legacy folder structure
train_model(
    train_folders=['./data/train/images'],
    val_folders=['./data/val/images'],
    batch_size=32,
    epochs=50
)
```

## Self-Supervised Learning

The model uses contrastive learning (InfoNCE loss) where:
- Two augmented views of the same image are created
- The model learns to produce similar representations for augmented views of the same image
- Different images should have different representations

## Modules

### DataLoader (`dataloader.py`)

The dataloader module provides two dataset classes:

1. **`DDTI`**: DDTI-style dataset class (based on [TRFE-Net](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation))
   - Expects directory structure: `root/image/` containing image files
   - Returns dictionary samples with 'image' key
   - Supports grayscale images (converted to 3 channels for DinoV2)

2. **`UltrasoundImageDataset`**: Legacy folder-based dataset class
   - Loads images from specified folders recursively
   - Supports common image formats (jpg, png, bmp, tiff)

### Fine-Tuning (`fine_tune_dinov2.py`)

The `DinoV2FineTuner` class wraps the DinoV2 model with a projection head for contrastive learning. The model can be fine-tuned end-to-end or with a frozen backbone.

### Visualization (`visualize_data.py`)

The visualization module provides functions to inspect and visualize data from dataloaders:

- `visualize_batch()`: Display a batch of images
- `visualize_augmentations()`: Compare original and augmented images
- `visualize_dataset_statistics()`: Show pixel value distributions and statistics
- `visualize_sample_images()`: Display specific images by index
- `print_dataloader_info()`: Print information about the dataloader

**Example usage in Jupyter notebook:**
```python
from visualize_data import (
    visualize_batch,
    visualize_augmentations,
    visualize_dataset_statistics,
    visualize_sample_images,
    print_dataloader_info
)
from dataloader import create_ddti_dataloader

# Create dataloader
dataloader = create_ddti_dataloader(
    root='/path/to/data/train',
    batch_size=4,
    shuffle=False,
    num_workers=0  # Use 0 for visualization in notebooks
)

# Print dataloader information
print_dataloader_info(dataloader)

# Visualize a batch
visualize_batch(dataloader, num_samples=8)

# Visualize dataset statistics
visualize_dataset_statistics(dataloader, num_batches=10)

# Visualize specific samples
visualize_sample_images(dataloader, indices=[0, 5, 10, 15])
```