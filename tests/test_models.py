"""
Basic tests for ViT models
"""

import torch
import sys
import os

# Add the parent directory to the path so we can import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_vit import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224
from models.deit import deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224
from models.swin_transformer import swin_tiny_patch4_window7_224


def test_model_creation():
    """Test that all models can be created successfully"""
    print("Testing model creation...")
    
    models = {
        'vit_tiny': vit_tiny_patch16_224,
        'vit_small': vit_small_patch16_224,
        'vit_base': vit_base_patch16_224,
        'deit_tiny': deit_tiny_patch16_224,
        'deit_small': deit_small_patch16_224,
        'deit_base': deit_base_patch16_224,
        'swin_tiny': swin_tiny_patch4_window7_224,
    }
    
    for name, model_fn in models.items():
        try:
            model = model_fn(num_classes=1000)
            print(f"  ✓ {name}: Created successfully")
        except Exception as e:
            print(f"  ✗ {name}: Failed - {e}")
            return False
    
    return True


def test_forward_pass():
    """Test forward pass for all models"""
    print("\nTesting forward pass...")
    
    # Test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    models = {
        'vit_tiny': vit_tiny_patch16_224(num_classes=10),
        'vit_small': vit_small_patch16_224(num_classes=10),
        'deit_tiny': deit_tiny_patch16_224(num_classes=10),
        'swin_tiny': swin_tiny_patch4_window7_224(num_classes=10),
    }
    
    for name, model in models.items():
        try:
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                
                # Handle different output formats
                if isinstance(output, tuple):
                    # DeiT during training
                    assert len(output) == 2, f"Expected tuple of length 2, got {len(output)}"
                    output = output[0]
                
                expected_shape = (batch_size, 10)
                assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
                
                print(f"  ✓ {name}: Forward pass successful, output shape {output.shape}")
                
        except Exception as e:
            print(f"  ✗ {name}: Forward pass failed - {e}")
            return False
    
    return True


def test_different_input_sizes():
    """Test models with different input configurations"""
    print("\nTesting different input configurations...")
    
    # Test different batch sizes
    for batch_size in [1, 4]:
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        model = vit_tiny_patch16_224(num_classes=5)
        model.eval()
        
        try:
            with torch.no_grad():
                output = model(input_tensor)
                expected_shape = (batch_size, 5)
                assert output.shape == expected_shape
                print(f"  ✓ Batch size {batch_size}: OK")
        except Exception as e:
            print(f"  ✗ Batch size {batch_size}: Failed - {e}")
            return False
    
    return True


def test_deit_training_mode():
    """Test DeiT specific behavior in training vs eval mode"""
    print("\nTesting DeiT training/eval modes...")
    
    model = deit_tiny_patch16_224(num_classes=10)
    input_tensor = torch.randn(1, 3, 224, 224)
    
    try:
        # Training mode - should return tuple
        model.train()
        with torch.no_grad():
            output_train = model(input_tensor)
            assert isinstance(output_train, tuple), "Training mode should return tuple"
            assert len(output_train) == 2, "Should return (cls_output, dist_output)"
            print("  ✓ Training mode: Returns tuple as expected")
        
        # Eval mode - should return single tensor
        model.eval()
        with torch.no_grad():
            output_eval = model(input_tensor)
            assert isinstance(output_eval, torch.Tensor), "Eval mode should return tensor"
            assert output_eval.shape == (1, 10), f"Expected (1, 10), got {output_eval.shape}"
            print("  ✓ Eval mode: Returns single tensor as expected")
        
        return True
        
    except Exception as e:
        print(f"  ✗ DeiT mode test failed: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction capabilities"""
    print("\nTesting feature extraction...")
    
    model = vit_small_patch16_224(num_classes=1000)
    input_tensor = torch.randn(1, 3, 224, 224)
    
    try:
        model.eval()
        with torch.no_grad():
            features = model.forward_features(input_tensor)
            
            # Expected: [batch_size, num_patches + 1, embed_dim]
            # +1 for CLS token
            expected_patches = (224 // 16) * (224 // 16)  # 196 patches for 224x224 with 16x16 patches
            expected_shape = (1, expected_patches + 1, 384)  # 384 is embed_dim for small model
            
            assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"
            print(f"  ✓ Feature extraction: Shape {features.shape}")
            
        return True
        
    except Exception as e:
        print(f"  ✗ Feature extraction failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("Running ViT Models Tests")
    print("=" * 40)
    
    tests = [
        test_model_creation,
        test_forward_pass,
        test_different_input_sizes,
        test_deit_training_mode,
        test_feature_extraction,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! ✓")
        return True
    else:
        print("Some tests failed! ✗")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)