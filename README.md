# ViT_Applications

A comprehensive collection of Vision Transformer (ViT) implementations for image classification and computer vision tasks. This repository provides clean, modular implementations of different ViT architectures with examples and training scripts.

## 🚀 Features

- **Multiple ViT Architectures**: Standard ViT, DeiT, and Swin Transformer implementations
- **Modular Design**: Easy to extend and modify for custom use cases
- **Training Examples**: Complete training scripts with dummy data
- **Model Comparison**: Utilities to compare different architectures
- **Feature Extraction**: Support for extracting intermediate features
- **Well Documented**: Comprehensive documentation and examples

## 🏗️ Architecture Implementations

### 1. **Vision Transformer (ViT)**
- **Paper**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- **Features**: Standard transformer architecture adapted for images
- **Variants**: Tiny, Small, Base, Large
- **Key Components**: Patch embedding, positional encoding, transformer blocks

### 2. **DeiT (Data-efficient Image Transformers)**
- **Paper**: "Training data-efficient image transformers & distillation through attention"
- **Features**: Knowledge distillation with teacher-student training
- **Variants**: Tiny, Small, Base with distillation token
- **Key Components**: Distillation token, dual classification heads

### 3. **Swin Transformer**
- **Paper**: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
- **Features**: Hierarchical architecture with shifted window attention
- **Variants**: Tiny, Small, Base
- **Key Components**: Window attention, patch merging, hierarchical feature maps

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/PawixTheGreat267/ViT_Applications.git
cd ViT_Applications
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision
- numpy
- pillow
- matplotlib
- timm
- einops

## 🚦 Quick Start

### Basic Usage

```python
from models.base_vit import vit_base_patch16_224
from models.deit import deit_base_patch16_224
from models.swin_transformer import swin_tiny_patch4_window7_224

# Create models
vit_model = vit_base_patch16_224(num_classes=1000)
deit_model = deit_base_patch16_224(num_classes=1000)
swin_model = swin_tiny_patch4_window7_224(num_classes=1000)

# Inference
import torch
input_tensor = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    vit_output = vit_model(input_tensor)
    deit_output = deit_model(input_tensor)
    swin_output = swin_model(input_tensor)
```

### Model Comparison

```bash
python examples/compare_models.py
```

### Inference Demo

```bash
python examples/inference_demo.py
```

### Training Example

```bash
python examples/train_vit.py
```

## 📊 Model Specifications

| Model | Parameters | Input Size | Patch Size | Embedding Dim | Layers | Heads |
|-------|------------|------------|------------|---------------|---------|--------|
| ViT-Ti/16 | 5.7M | 224×224 | 16×16 | 192 | 12 | 3 |
| ViT-S/16 | 22.1M | 224×224 | 16×16 | 384 | 12 | 6 |
| ViT-B/16 | 86.6M | 224×224 | 16×16 | 768 | 12 | 12 |
| DeiT-Ti/16 | 5.7M | 224×224 | 16×16 | 192 | 12 | 3 |
| DeiT-S/16 | 22.1M | 224×224 | 16×16 | 384 | 12 | 6 |
| DeiT-B/16 | 86.6M | 224×224 | 16×16 | 768 | 12 | 12 |
| Swin-T | 28.3M | 224×224 | 4×4 | 96 | [2,2,6,2] | [3,6,12,24] |

## 📁 Project Structure

```
ViT_Applications/
├── models/
│   ├── __init__.py          # Model exports
│   ├── base_vit.py          # Standard Vision Transformer
│   ├── deit.py              # DeiT implementation
│   ├── swin_transformer.py  # Swin Transformer
│   └── utils.py             # Shared utilities
├── examples/
│   ├── compare_models.py    # Model comparison script
│   ├── inference_demo.py    # Inference demonstration
│   └── train_vit.py         # Training example
├── tests/
│   └── test_models.py       # Unit tests
├── data/
│   └── __init__.py          # Data utilities
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🧪 Testing

Run the test suite to verify all models work correctly:

```bash
python tests/test_models.py
```

The tests verify:
- Model creation and initialization
- Forward pass with different batch sizes
- DeiT training/evaluation mode behavior
- Feature extraction capabilities
- Output shape consistency

## 💡 Usage Examples

### Feature Extraction

```python
from models.base_vit import vit_base_patch16_224

model = vit_base_patch16_224(num_classes=1000)
model.eval()

# Extract features before classification
with torch.no_grad():
    features = model.forward_features(input_tensor)
    # features shape: [batch_size, num_patches + 1, embed_dim]
    
    cls_token = features[:, 0]  # Classification token
    patch_features = features[:, 1:]  # Patch tokens
```

### Custom Number of Classes

```python
# For custom datasets
model = vit_base_patch16_224(num_classes=10)  # CIFAR-10
model = deit_base_patch16_224(num_classes=100)  # CIFAR-100
model = swin_tiny_patch4_window7_224(num_classes=200)  # Custom dataset
```

### DeiT Distillation Training

```python
from models.deit import deit_base_patch16_224

model = deit_base_patch16_224(num_classes=1000)

# Training mode returns both classification and distillation outputs
model.train()
output_cls, output_dist = model(input_tensor)

# Evaluation mode returns averaged output
model.eval()
output = model(input_tensor)
```

## 🔬 Model Details

### Vision Transformer (ViT)
- Uses patch embedding to convert images into sequences
- Employs standard transformer encoder blocks
- Uses [CLS] token for final classification
- Supports different patch sizes and model dimensions

### DeiT (Data-efficient Image Transformers)
- Extends ViT with knowledge distillation
- Includes distillation token alongside [CLS] token
- Dual classification heads for student-teacher learning
- More efficient training with less data

### Swin Transformer
- Hierarchical transformer with shifted windows
- More efficient for high-resolution images
- Window-based self-attention reduces computational complexity
- Produces multi-scale feature representations

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new features or models
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

This implementation is based on the following papers:

1. **ViT**: Dosovitskiy, A., et al. "An image is worth 16x16 words: Transformers for image recognition at scale." ICLR 2021.

2. **DeiT**: Touvron, H., et al. "Training data-efficient image transformers & distillation through attention." ICML 2021.

3. **Swin**: Liu, Z., et al. "Swin transformer: Hierarchical vision transformer using shifted windows." ICCV 2021.

## 📚 Citation

If you use this code in your research, please cite the original papers and this repository:

```bibtex
@misc{vit_applications,
  title={ViT Applications: Multiple Vision Transformer Implementations},
  author={PawixTheGreat267},
  year={2024},
  url={https://github.com/PawixTheGreat267/ViT_Applications}
}
```