"""Custom enhancement modules for Ultralytics models.

This file contains a drop-in, YAML-friendly version of AdaptivePerChannelGamma
adapted to the model parser expectations (i.e. constructors accept c1, c2, ...).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePerChannelGamma(nn.Module):
    """Safe identity module for testing - pure pass-through."""

    def __init__(self, c1: int, c2: int = None, num_channels: int = None):
        super().__init__()
        # No parameters at all - pure identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # Always pass through unchanged


class LearnableCLAHE(nn.Module):
    """Differentiable, learnable CLAHE-like module compatible with YAML parser.

    Simplified stable version with minimal processing.
    """

    def __init__(self, c1: int, c2: int = None, grid_size=(8, 8)):
        super().__init__()
        ch = c2 if c2 is not None else c1
        
        # Simple learnable parameters (initialized conservatively)
        self.contrast = nn.Parameter(torch.ones(ch) * 0.98)
        self.brightness = nn.Parameter(torch.zeros(ch))
        self.enabled = False  # Can be set to True after initial training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x  # Pass through during initial training
            
        # Simple contrast/brightness adjustment with residual
        contrast_clamped = torch.clamp(self.contrast, 0.9, 1.1)
        brightness_clamped = torch.clamp(self.brightness, -0.1, 0.1)
        
        enhanced = x * contrast_clamped.view(1, -1, 1, 1) + brightness_clamped.view(1, -1, 1, 1)
        return enhanced


class TrainableRetinex(nn.Module):
    """Trainable Retinex-style enhancement compatible with YAML parser.

    Simplified stable version with minimal processing.
    """

    def __init__(self, c1: int, c2: int = None, kernel_size: int = 15, sigma: float = 5.0):
        super().__init__()
        ch = c2 if c2 is not None else c1
        
        # Simple learnable weight per channel (initialized near 1.0)
        self.weight = nn.Parameter(torch.ones(ch) * 0.99)
        self.enabled = False  # Can be set to True after initial training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x  # Pass through during initial training
            
        # Simple channel-wise weighting
        weight_clamped = torch.clamp(self.weight, 0.9, 1.1)
        return x * weight_clamped.view(1, -1, 1, 1)
