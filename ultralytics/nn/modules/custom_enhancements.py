"""Custom enhancement modules for Ultralytics models.

This file contains YAML-friendly versions of enhancement modules adapted to
the model parser expectations (i.e. constructors accept c1, c2, ...).
Includes PSM (Dynamic Feature Extraction Module) from DarkYOLO paper.
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


class CSPPF(nn.Module):
    """Cross-Spatial Pyramid Pooling Feature (CSPPF) module from DarkYOLO.
    
    Employs overlapping pooling strategy combining max and average pooling
    to preserve local information in low-light images. Uses two parallel pathways:
    - Pathway 1: Max -> Max -> Avg pooling sequence
    - Pathway 2: Avg -> Avg -> Max pooling sequence
    """
    
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        
        # If c1 != c2, we need a projection layer
        if c1 != c2:
            self.proj = nn.Conv2d(c1, c2, 1, bias=False)
        else:
            self.proj = None
        
        # Initial convolution (Equation 4: x = Conv(xinput))
        self.cv1 = nn.Conv2d(c2, c_, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        
        # Pooling operations for both pathways
        self.maxpool1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=k, stride=1, padding=k//2)
        
        # Final convolution to combine all concatenated features
        # Input channels: c_ (original) + 6*c_ (from 6 pooling operations) = 7*c_
        self.cv2 = nn.Conv2d(7 * c_, c2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        
    def forward(self, x):
        # Apply projection if needed
        if self.proj is not None:
            x = self.proj(x)
        
        # Initial convolution (Equation 4)
        x = self.act(self.bn1(self.cv1(x)))
        
        # Pathway 1: Max -> Max -> Avg sequence
        max1_x = self.maxpool1(x)                    # First max pooling
        max2_x = self.maxpool1(max1_x)               # Second max pooling  
        avg_after_max2 = self.avgpool1(max2_x)      # Average after two max
        
        # Pathway 2: Avg -> Avg -> Max sequence  
        avg1_x = self.avgpool1(x)                    # First average pooling
        avg2_x = self.avgpool1(avg1_x)               # Second average pooling
        max_after_avg2 = self.maxpool1(avg2_x)      # Max after two average
        
        # Concatenations as per equations 5, 6, 7
        # concat1 = concat(Max(x), Avg(x)) (Equation 5)
        concat1 = torch.cat([max1_x, avg1_x], dim=1)
        
        # concat2 = concat(Max(Max(x)), Avg(Avg(x))) (Equation 6)  
        concat2 = torch.cat([max2_x, avg2_x], dim=1)
        
        # concat3 = concat(Max(Max(Max(x))), Avg(Avg(Avg(x)))) (Equation 7)
        # Note: For the third level, we use the cross-pathway results
        concat3 = torch.cat([avg_after_max2, max_after_avg2], dim=1)
        
        # Final concatenation (Equation 8: y = concat(concat1, concat2, concat3))
        # Also include original x for residual-like connection
        y = torch.cat([x, concat1, concat2, concat3], dim=1)
        
        # Final convolution and activation
        return self.act(self.bn2(self.cv2(y)))


class SimAM(nn.Module):
    """Parameter-free SimAM attention mechanism from DarkYOLO PSM block.
    
    Generates 3D attention weights without additional parameters by evaluating
    local self-similarity of feature maps.
    """
    
    def __init__(self, lambda_reg=1e-4):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten spatial dimensions for easier computation
        x_flat = x.view(B, C, -1)  # [B, C, H*W]
        
        # Compute mean and variance per channel
        mu = torch.mean(x_flat, dim=2, keepdim=True)  # [B, C, 1]
        var = torch.var(x_flat, dim=2, keepdim=True)  # [B, C, 1]
        
        # Compute energy function for each neuron (Equation 9)
        # e_i = 4(σ + λ) / [(t_i - μ)² + 2σ² + 2λ]
        numerator = 4 * (var + self.lambda_reg)
        denominator = (x_flat - mu) ** 2 + 2 * var + 2 * self.lambda_reg
        energy = numerator / (denominator + 1e-8)  # Add small epsilon for stability
        
        # Generate attention weights (Equation 10)
        # X' = sigmoid(1/e_i) * X
        attention = torch.sigmoid(1.0 / (energy + 1e-8))
        attention = attention.view(B, C, H, W)
        
        return x * attention


class PartialConv2d(nn.Module):
    """Partial Convolution that processes only valid pixels.
    
    Applies convolution to only a subset of input channels while leaving
    others unchanged, reducing computational load for low-light conditions.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, bias=True, partial_ratio=0.5):
        super().__init__()
        self.partial_channels = max(1, int(in_channels * partial_ratio))
        self.remaining_channels = in_channels - self.partial_channels
        
        # Convolution applied only to partial channels
        self.partial_conv = nn.Conv2d(
            self.partial_channels, out_channels - self.remaining_channels,
            kernel_size, stride, padding, bias=bias
        )
        
    def forward(self, x):
        # Split input into partial and remaining channels
        x_partial = x[:, :self.partial_channels, :, :]
        x_remaining = x[:, self.partial_channels:, :, :]
        
        # Apply convolution only to partial channels
        x_partial_out = self.partial_conv(x_partial)
        
        # Concatenate processed partial channels with unchanged remaining channels
        return torch.cat([x_partial_out, x_remaining], dim=1)


class PSMBlock(nn.Module):
    """Dynamic Feature Extraction Module (PSM) from DarkYOLO.
    
    Integrates parameter-free SimAM attention with partial convolution
    for enhanced low-light feature extraction.
    """
    
    def __init__(self, c1, c2, n=1, partial_ratio=0.5, lambda_reg=1e-4):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        # If c1 != c2, we need a projection layer
        if c1 != c2:
            self.proj = nn.Conv2d(c1, c2, 1, bias=False)
        else:
            self.proj = None
        
        # Initial feature extraction (2D convolution)
        self.initial_conv = nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        
        # Split into multiple branches for feature extraction
        self.branch_channels = c2 // 4
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c2, self.branch_channels, 1, bias=False),
                nn.BatchNorm2d(self.branch_channels),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(c2, self.branch_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.branch_channels),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(c2, self.branch_channels, 3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(self.branch_channels),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(c2, self.branch_channels, 3, padding=3, dilation=3, bias=False),
                nn.BatchNorm2d(self.branch_channels),
                nn.SiLU()
            )
        ])
        
        # Parameter-free SimAM attention
        self.attention = SimAM(lambda_reg=lambda_reg)
        
        # Partial convolution for refinement
        self.partial_conv = PartialConv2d(
            c2, c2, kernel_size=3, padding=1, partial_ratio=partial_ratio
        )
        
        # Final fusion layer
        self.fusion_conv = nn.Conv2d(c2, c2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    
    def forward(self, x):
        # Apply projection if needed
        if self.proj is not None:
            x = self.proj(x)
        
        # Initial feature extraction
        x = self.act(self.bn1(self.initial_conv(x)))
        
        # Multi-branch feature extraction
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # Concatenate branch outputs
        multi_scale_features = torch.cat(branch_outputs, dim=1)
        
        # Apply SimAM attention mechanism
        attended_features = self.attention(multi_scale_features)
        
        # Apply partial convolution for refinement
        refined_features = self.partial_conv(attended_features)
        
        # Final fusion and residual connection
        output = self.act(self.bn2(self.fusion_conv(refined_features)))
        return output + x  # Residual connection
