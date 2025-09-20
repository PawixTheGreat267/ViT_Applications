"""
DeiT (Data-efficient Image Transformers) implementation
Based on "Training data-efficient image transformers & distillation through attention"
"""

import torch
import torch.nn as nn
from .base_vit import VisionTransformer, _init_vit_weights
from .utils import trunc_normal_


class DeiT(VisionTransformer):
    """DeiT model with distillation token"""
    
    def __init__(self, *args, **kwargs):
        # Force distilled=True for DeiT
        kwargs['distilled'] = True
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = self.forward_features(x)
        x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, 1])
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """DeiT-Tiny model (DeiT-Ti/16)"""
    model = DeiT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = {
        'url': '',
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
    }
    return model


def deit_small_patch16_224(pretrained=False, **kwargs):
    """DeiT-Small model (DeiT-S/16)"""
    model = DeiT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = {
        'url': '',
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
    }
    return model


def deit_base_patch16_224(pretrained=False, **kwargs):
    """DeiT-Base model (DeiT-B/16)"""
    model = DeiT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = {
        'url': '',
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
    }
    return model


def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """DeiT-Base Distilled model (DeiT-B/16)"""
    model = deit_base_patch16_224(pretrained=pretrained, **kwargs)
    return model