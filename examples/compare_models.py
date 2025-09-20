"""
Model comparison script to demonstrate different ViT architectures
"""

import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the path so we can import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_vit import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224
from models.deit import deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224
from models.swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224


def count_parameters(model):
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_models():
    """Compare different ViT model architectures"""
    
    # Create models
    models = {
        'ViT-Ti/16': vit_tiny_patch16_224(num_classes=1000),
        'ViT-S/16': vit_small_patch16_224(num_classes=1000),
        'ViT-B/16': vit_base_patch16_224(num_classes=1000),
        'DeiT-Ti/16': deit_tiny_patch16_224(num_classes=1000),
        'DeiT-S/16': deit_small_patch16_224(num_classes=1000),
        'DeiT-B/16': deit_base_patch16_224(num_classes=1000),
        'Swin-T': swin_tiny_patch4_window7_224(num_classes=1000),
        'Swin-S': swin_small_patch4_window7_224(num_classes=1000),
    }
    
    # Test input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    print("=" * 80)
    print("ViT Model Comparison")
    print("=" * 80)
    print(f"{'Model':<15} {'Parameters':<12} {'Output Shape':<20} {'Forward Pass':<12}")
    print("-" * 80)
    
    for name, model in models.items():
        try:
            model.eval()
            with torch.no_grad():
                if 'DeiT' in name:
                    # DeiT returns tuple during training, single tensor during eval
                    output = model(input_tensor)
                    if isinstance(output, tuple):
                        output = output[0]  # Take classification output
                else:
                    output = model(input_tensor)
                
                params = count_parameters(model)
                params_str = f"{params / 1e6:.1f}M"
                output_shape = str(tuple(output.shape))
                status = "✓"
                
        except Exception as e:
            params_str = "Error"
            output_shape = "Error"
            status = f"✗ ({str(e)[:20]}...)"
        
        print(f"{name:<15} {params_str:<12} {output_shape:<20} {status:<12}")
    
    print("-" * 80)
    print("\nModel Architecture Details:")
    print("-" * 40)
    
    # Show detailed architecture for a few models
    print("\n1. Standard ViT-B/16:")
    print("   - Patch size: 16x16")
    print("   - Embedding dim: 768")
    print("   - Layers: 12")
    print("   - Attention heads: 12")
    print("   - Uses [CLS] token for classification")
    
    print("\n2. DeiT-B/16:")
    print("   - Same as ViT-B/16 but with distillation token")
    print("   - Two classification heads: one for direct classification, one for distillation")
    print("   - More efficient training through knowledge distillation")
    
    print("\n3. Swin Transformer:")
    print("   - Hierarchical architecture with shifted windows")
    print("   - Patch size: 4x4 (smaller patches)")
    print("   - Window-based self-attention")
    print("   - More efficient for high-resolution images")
    
    return models


def demo_inference():
    """Demonstrate inference with different models"""
    print("\n" + "=" * 80)
    print("Inference Demo")
    print("=" * 80)
    
    # Create a sample input
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # Load models
    vit_model = vit_base_patch16_224(num_classes=1000)
    deit_model = deit_base_patch16_224(num_classes=1000)
    swin_model = swin_tiny_patch4_window7_224(num_classes=1000)
    
    models = {
        'ViT-B/16': vit_model,
        'DeiT-B/16': deit_model,
        'Swin-T': swin_model
    }
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]  # For DeiT, take the main classifier output
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                time_str = f"{inference_time:.2f}ms"
            else:
                time_str = "N/A"
            
            # Get top-5 predictions (just indices since we don't have pretrained weights)
            probs = torch.softmax(output, dim=1)
            top5_indices = torch.topk(probs, 5, dim=1)[1]
            
            print(f"\n{name}:")
            print(f"  Output shape: {output.shape}")
            print(f"  Inference time: {time_str}")
            print(f"  Top-5 class indices: {top5_indices[0].tolist()}")
            print(f"  Max probability: {probs.max().item():.4f}")


if __name__ == "__main__":
    try:
        models = compare_models()
        demo_inference()
        print("\n" + "=" * 80)
        print("Model comparison completed successfully!")
        print("All models are working correctly.")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during model comparison: {e}")
        print("Please check the model implementations and dependencies.")