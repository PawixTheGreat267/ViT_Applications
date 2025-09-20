"""
Simple inference demo for ViT models
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os
import sys

# Add the parent directory to the path so we can import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_vit import vit_base_patch16_224
from models.deit import deit_base_patch16_224
from models.swin_transformer import swin_tiny_patch4_window7_224


def get_image_transforms():
    """Get standard ImageNet transforms"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_dummy_image():
    """Create a dummy image for demonstration"""
    # Create a simple gradient image
    import numpy as np
    
    # Create RGB gradient
    height, width = 224, 224
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Red gradient
    image[:, :, 0] = np.linspace(0, 255, width).astype(np.uint8)
    # Green gradient  
    image[:, :, 1] = np.linspace(0, 255, height).reshape(-1, 1).astype(np.uint8)
    # Blue constant
    image[:, :, 2] = 128
    
    return Image.fromarray(image)


def run_inference_demo():
    """Run inference demo with different ViT models"""
    print("ViT Models Inference Demo")
    print("=" * 50)
    
    # Create models
    models = {
        'ViT-Base': vit_base_patch16_224(num_classes=1000),
        'DeiT-Base': deit_base_patch16_224(num_classes=1000),
        'Swin-Tiny': swin_tiny_patch4_window7_224(num_classes=1000),
    }
    
    # Create dummy input
    image = create_dummy_image()
    transform = get_image_transforms()
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input statistics: mean={input_tensor.mean():.3f}, std={input_tensor.std():.3f}")
    print()
    
    # Run inference for each model
    for model_name, model in models.items():
        print(f"Running inference with {model_name}...")
        
        model.eval()
        with torch.no_grad():
            try:
                # Forward pass
                output = model(input_tensor)
                
                # Handle DeiT output (tuple during training)
                if isinstance(output, tuple):
                    output = output[0]
                
                # Get predictions
                probabilities = F.softmax(output, dim=1)
                top5_probs, top5_indices = torch.topk(probabilities, 5)
                
                print(f"  Output shape: {output.shape}")
                print(f"  Top 5 predictions:")
                for i, (idx, prob) in enumerate(zip(top5_indices[0], top5_probs[0])):
                    print(f"    {i+1}. Class {idx.item():4d}: {prob.item():.4f}")
                
                # Model statistics
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"  Model parameters: {total_params:,} ({trainable_params:,} trainable)")
                print()
                
            except Exception as e:
                print(f"  Error: {e}")
                print()
    
    return models


def model_feature_extraction():
    """Demonstrate feature extraction capabilities"""
    print("Feature Extraction Demo")
    print("=" * 50)
    
    # Create a model for feature extraction
    model = vit_base_patch16_224(num_classes=1000)
    model.eval()
    
    # Create dummy input
    image = create_dummy_image()
    transform = get_image_transforms()
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        # Extract features before the classification head
        features = model.forward_features(input_tensor)
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Feature shape: {features.shape}")
        print(f"CLS token features: {features[0, 0, :5]}")  # First 5 dimensions of CLS token
        print(f"Patch features shape: {features[0, 1:, :].shape}")  # All patch tokens
        print()
        
        # Show attention patterns (simplified)
        print("Model Architecture Summary:")
        print(f"- Patch embedding dimension: {model.embed_dim}")
        print(f"- Number of transformer blocks: {len(model.blocks)}")
        print(f"- Number of attention heads: {model.blocks[0].attn.num_heads}")
        print(f"- Number of patches: {model.patch_embed.num_patches}")


if __name__ == "__main__":
    try:
        print("Starting ViT inference demo...\n")
        
        # Run main inference demo
        models = run_inference_demo()
        
        # Feature extraction demo
        model_feature_extraction()
        
        print("Demo completed successfully!")
        print("\nNote: These models are randomly initialized.")
        print("For real applications, you would load pretrained weights.")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install torch torchvision pillow")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()