"""
Training example for ViT models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# Add the parent directory to the path so we can import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_vit import vit_small_patch16_224
from models.deit import deit_small_patch16_224


def create_dummy_dataset(num_samples=1000, num_classes=10):
    """Create a dummy dataset for training demonstration"""
    
    # Generate random images
    images = torch.randn(num_samples, 3, 224, 224)
    
    # Generate random labels
    labels = torch.randint(0, num_classes, (num_samples,))
    
    return TensorDataset(images, labels)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Handle DeiT output (returns tuple during training)
        if isinstance(output, tuple):
            output_cls, output_dist = output
            # Use both outputs for loss calculation (simplified distillation)
            loss = 0.5 * criterion(output_cls, target) + 0.5 * criterion(output_dist, target)
            output = output_cls  # Use classification output for accuracy
        else:
            loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx:3d}: Loss={loss.item():.4f}, '
                  f'Acc={100.*correct/total:.1f}% ({correct}/{total})')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Handle DeiT output
            if isinstance(output, tuple):
                output = output[0]  # Use classification output
            
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = test_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def training_demo():
    """Demonstrate training of ViT models"""
    print("ViT Training Demo")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    num_classes = 10
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 3
    
    # Create dataset
    print("\nCreating dummy dataset...")
    dataset = create_dummy_dataset(num_samples=200, num_classes=num_classes)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Test different models
    models_to_test = {
        'ViT-Small': vit_small_patch16_224(num_classes=num_classes),
        'DeiT-Small': deit_small_patch16_224(num_classes=num_classes),
    }
    
    for model_name, model in models_to_test.items():
        print(f"\n" + "=" * 30)
        print(f"Training {model_name}")
        print("=" * 30)
        
        model = model.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 20)
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%")
        
        # Model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")


def demonstrate_model_features():
    """Demonstrate special features of different models"""
    print("\n" + "=" * 50)
    print("Model Features Demonstration")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample input
    input_tensor = torch.randn(2, 3, 224, 224).to(device)
    
    print("\n1. Standard ViT features:")
    vit_model = vit_small_patch16_224(num_classes=10).to(device)
    vit_model.eval()
    
    with torch.no_grad():
        features = vit_model.forward_features(input_tensor)
        output = vit_model(input_tensor)
        
        print(f"   Features shape: {features.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Number of patches: {vit_model.patch_embed.num_patches}")
    
    print("\n2. DeiT features (with distillation):")
    deit_model = deit_small_patch16_224(num_classes=10).to(device)
    deit_model.train()  # Show training behavior
    
    with torch.no_grad():
        features = deit_model.forward_features(input_tensor)
        output = deit_model(input_tensor)
        
        print(f"   Features shape: {features.shape}")
        if isinstance(output, tuple):
            print(f"   Classification output shape: {output[0].shape}")
            print(f"   Distillation output shape: {output[1].shape}")
            print("   DeiT returns two outputs during training for knowledge distillation")
        else:
            print(f"   Output shape: {output.shape}")
    
    # Switch to eval mode
    deit_model.eval()
    with torch.no_grad():
        output_eval = deit_model(input_tensor)
        print(f"   Eval mode output shape: {output_eval.shape}")
        print("   In eval mode, DeiT averages both classifier outputs")


if __name__ == "__main__":
    try:
        training_demo()
        demonstrate_model_features()
        
        print("\n" + "=" * 50)
        print("Training demo completed successfully!")
        print("Note: This is a simplified training example with dummy data.")
        print("For real training, use proper datasets like ImageNet.")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during training demo: {e}")
        import traceback
        traceback.print_exc()