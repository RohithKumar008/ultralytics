"""Custom enhancement modules for Ultralytics models.

This file contains YAML-friendly versions of enhancement modules adapted to
the model parser expectations (i.e. constructors accept c1, c2, ...).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptivePerChannelGamma(nn.Module):
    """Adaptive Per-Channel Gamma Correction for YOLO integration."""

    def __init__(self, c1: int, c2: int = None, *args, **kwargs):
        super().__init__()
        # If c2 not specified, use c1 (true identity)
        if c2 is None:
            c2 = c1
        
        self.c1 = c1
        self.c2 = c2
        
        # If c1 != c2, we need a projection layer
        if c1 != c2:
            self.proj = nn.Conv2d(c1, c2, 1, bias=False)
        else:
            self.proj = None
        
        # Simplified approach: Use global average pooling + per-channel learnable gamma
        # This avoids the dimension mismatch issue entirely
        self.base_gamma = nn.Parameter(torch.ones(c2))
        self.adaptive_weight = nn.Parameter(torch.ones(c2) * 0.1)  # Small adaptive component

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply projection if needed
        if self.proj is not None:
            x = self.proj(x)
        
        # Simple adaptive gamma based on global average per channel
        global_avg = torch.mean(x, dim=(2, 3), keepdim=False)  # [B, C]
        global_avg = torch.mean(global_avg, dim=0)  # [C] - average across batch
        
        # Compute adaptive gamma (simpler and more stable)
        adaptive_factor = torch.sigmoid(self.adaptive_weight * global_avg)
        gamma = torch.clamp(self.base_gamma * (0.5 + adaptive_factor), 0.5, 2.0)
        
        # Apply gamma correction
        out = torch.pow(x + 1e-6, gamma.view(1, -1, 1, 1))
        return out


class LearnableCLAHE(nn.Module):
    """Differentiable, learnable CLAHE-like module compatible with YAML parser."""

    def __init__(self, c1: int, c2: int = None, grid_size=(8, 8), *args, **kwargs):
        super().__init__()
        # If c2 not specified, use c1
        if c2 is None:
            c2 = c1
        
        self.c1 = c1
        self.c2 = c2
        self.grid_size = grid_size
        
        # If c1 != c2, we need a projection layer
        if c1 != c2:
            self.proj = nn.Conv2d(c1, c2, 1, bias=False)
        else:
            self.proj = None
        
        # Learnable "contrast strength" per channel (using c2)
        self.alpha = nn.Parameter(torch.ones(c2))  # controls amplification
        self.beta = nn.Parameter(torch.zeros(c2))  # controls brightness offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply projection if needed
        if self.proj is not None:
            x = self.proj(x)
        
        B, C, H, W = x.shape
        # Normalize [0,1]
        x = torch.clamp(x, 0, 1)
        
        # Compute local mean & std using adaptive average pooling
        mean = F.adaptive_avg_pool2d(x, self.grid_size)
        std = torch.sqrt(F.adaptive_avg_pool2d((x - F.interpolate(mean, (H, W)))**2, self.grid_size) + 1e-6)
        mean_up = F.interpolate(mean, (H, W), mode='bilinear', align_corners=False)
        std_up = F.interpolate(std, (H, W), mode='bilinear', align_corners=False)
        
        # CLAHE-like normalization (differentiable)
        enhanced = (x - mean_up) / (std_up + 1e-6)
        enhanced = enhanced * self.alpha.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        enhanced = torch.clamp(enhanced * 0.5 + 0.5, 0, 1)  # map back to [0,1]
        return enhanced


class TrainableRetinex(nn.Module):
    """Trainable Retinex-style enhancement compatible with YAML parser."""

    def __init__(self, c1: int, c2: int = None, kernel_size: int = 15, sigma: float = 5.0, *args, **kwargs):
        super().__init__()
        # If c2 not specified, use c1
        if c2 is None:
            c2 = c1
        
        self.c1 = c1
        self.c2 = c2
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.eps = 1e-6
        
        # If c1 != c2, we need a projection layer
        if c1 != c2:
            self.proj = nn.Conv2d(c1, c2, 1, bias=False)
        else:
            self.proj = None

        # --- Learnable alpha per channel (using c2) ---
        self.alpha = nn.Parameter(torch.ones(c2))  # starts as [1,1,1,...]
        
        # --- Fixed Gaussian kernel (not learnable) ---
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel(kernel_size, sigma))
    
    def _create_gaussian_kernel(self, ksize, sigma):
        ax = torch.arange(-ksize // 2 + 1., ksize // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, ksize, ksize)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply projection if needed
        if self.proj is not None:
            x = self.proj(x)
        
        B, C, H, W = x.shape
        out = []
        for c in range(C):
            I = x[:, c:c+1, :, :]
            kernel = self.gaussian_kernel.expand(B, 1, -1, -1)
            blur = F.conv2d(I, kernel, padding=self.kernel_size // 2, groups=1)
            retinex = torch.log(I + self.eps) - torch.log(blur + self.eps)
            enhanced = self.alpha[c] * retinex
            out.append(enhanced)
        out = torch.cat(out, dim=1)
        return torch.clamp(out, -2, 2)  # optional normalization
