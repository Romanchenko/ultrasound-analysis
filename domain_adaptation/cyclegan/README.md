# CycleGAN for Domain Adaptation

This directory contains the implementation of CycleGAN for unpaired image-to-image translation between different ultrasound image domains.

## Structure

```
cyclegan/
├── model.py              # CycleGAN model definition (generators, discriminators)
├── train.py              # Training loop and trainer class
├── dataloaders/
│   ├── __init__.py
│   ├── base_dataloader.py    # Base classes for custom dataloaders
│   └── example_dataloader.py  # Example implementation
└── README.md
```

## Model Architecture

The CycleGAN model consists of:

- **Two Generators**: `G_A2B` (domain A → B) and `G_B2A` (domain B → A)
- **Two Discriminators**: `D_A` (discriminates domain A) and `D_B` (discriminates domain B)

### Generator Architecture
- ResNet-based architecture with residual blocks
- Uses instance normalization
- 9 residual blocks by default (configurable)

### Discriminator Architecture
- PatchGAN architecture
- 70x70 patch size
- Uses instance normalization

## Usage

### Basic Usage

```python
from domain_adaptation.cyclegan import create_cyclegan_model, train_cyclegan
from domain_adaptation.cyclegan.dataloaders.example_dataloader import create_example_dataloader

# Create model
model = create_cyclegan_model(
    input_channels_a=1,  # Grayscale images
    input_channels_b=1,
    n_residual_blocks=9
)

# Create dataloader
train_loader = create_example_dataloader(
    root_a='path/to/domain_a/images',
    root_b='path/to/domain_b/images',
    batch_size=1,
    image_size=256
)

# Train
trainer = train_cyclegan(
    model=model,
    train_loader=train_loader,
    n_epochs=200,
    checkpoint_dir='./checkpoints/cyclegan'
)
```

### Creating Custom Dataloaders

To create a custom dataloader for your specific dataset:

1. **Subclass `UnpairedDataset`** from `base_dataloader.py`:

```python
from domain_adaptation.cyclegan.dataloaders.base_dataloader import UnpairedDataset
from pathlib import Path
from PIL import Image

class MyCustomDataset(UnpairedDataset):
    def _get_image_paths(self, root: Path):
        # Implement your logic to get image paths
        # e.g., from a CSV file, specific folder structure, etc.
        return list(root.glob('*.png'))
    
    def _load_image(self, path: Path):
        # Implement your image loading logic
        # e.g., handle specific image formats, preprocessing, etc.
        return Image.open(path).convert('RGB')
```

2. **Create a factory function**:

```python
from domain_adaptation.cyclegan.dataloaders.base_dataloader import create_dataloader
from torchvision import transforms

def create_my_custom_dataloader(root_a, root_b, batch_size=1, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = MyCustomDataset(
        root_a=root_a,
        root_b=root_b,
        transform_a=transform,
        transform_b=transform
    )
    
    return create_dataloader(dataset, batch_size=batch_size)
```

## Training Parameters

- **lambda_cycle**: Weight for cycle consistency loss (default: 10.0)
- **lambda_identity**: Weight for identity loss (default: 0.5)
- **lr_g**: Learning rate for generators (default: 2e-4)
- **lr_d**: Learning rate for discriminators (default: 2e-4)

## Loss Functions

The model uses three types of losses:

1. **Adversarial Loss**: Ensures generated images are realistic
2. **Cycle Consistency Loss**: Ensures A→B→A reconstruction matches original A
3. **Identity Loss**: Helps preserve color/tone when translating similar images

## Checkpointing

Checkpoints are saved automatically:
- Every `save_every` epochs (default: 10)
- Best model (lowest generator loss) is saved as `best_model.pt`

## Notes

- The model expects images normalized to [-1, 1] range
- Default batch size is 1 (can be increased if memory allows)
- Training is typically done for 200+ epochs
- Instance normalization is used instead of batch normalization for better results on small batches

