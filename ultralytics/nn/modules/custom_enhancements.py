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


class SCINet(nn.Module):
    """SCINet: Simple YAML-friendly implementation of the SCI low-light enhancement module.

    Constructor signature matches YAML-friendly pattern used across Ultralytics modules:
        SCINet(c1, c2=None, base_ch=32, stages=4, share_weights=True)

    - c1: input channels (from parser)
    - c2: output channels (optional). If provided and different, a 1x1 proj is applied.
    - base_ch: internal channel width for the lightweight enhancement blocks
    - stages: number of cascaded illumination stages
    - share_weights: whether to reuse the same stage (weight sharing)
    """

    def __init__(self, c1, c2=None, base_ch=32, stages=4, share_weights=True, *args, **kwargs):
        super().__init__()
        if c2 is None:
            c2 = c1
        self.c1 = c1
        self.c2 = c2
        # projection if channel mismatch
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else None

        # Lightweight building blocks
        def make_stage(in_ch, base_ch_inner=base_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, base_ch_inner, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch_inner),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_ch_inner, base_ch_inner, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch_inner),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_ch_inner, in_ch, 1, bias=True),
            )

        # self-calibrated module (kappa) and illumination estimator (H)
        class Stage(nn.Module):
            def __init__(self, ch, base_ch_local):
                super().__init__()
                self.kappa = make_stage(ch, base_ch_local)
                self.H = make_stage(ch, base_ch_local)

            def forward(self, x_t, y):
                # z = y / (x + eps)
                z = y / (x_t + 1e-6)
                s = torch.tanh(self.kappa(z)) * 0.2
                v = y + s
                u = torch.tanh(self.H(v)) * 0.1
                x_next = x_t + u
                return x_next, v, s, u

        if share_weights:
            self.stage = Stage(c2, base_ch)
            self.stages = stages
        else:
            self.stage_list = nn.ModuleList([Stage(c2, base_ch) for _ in range(stages)])
            self.stages = stages
        self.share_weights = share_weights

    def forward(self, x0, y=None):
        """Run cascaded SCI enhancement.

        Args:
            x0: initial illumination estimate tensor [B,C,H,W]
            y: input low-light image tensor [B,C,H,W]. If None, x0 is used as input too.
        Returns:
            dict with keys: x_list (list of x_t), s_list, u_list, v_list, enhanced
        """
        # If a projection was created to match channels, apply it to inputs
        if self.proj is not None:
            # project x0 and y (if present) to the internal channel size
            x0 = self.proj(x0)
            if y is not None:
                y = self.proj(y)

        if y is None:
            # assume x0 is the input image
            y = x0
            x_t = x0.clone()
        else:
            x_t = x0 if x0 is not None else y.clone()

        x_list, s_list, u_list, v_list = [], [], [], []
        for t in range(self.stages):
            if self.share_weights:
                x_t, v, s, u = self.stage(x_t, y)
            else:
                x_t, v, s, u = self.stage_list[t](x_t, y)
            x_list.append(x_t)
            s_list.append(s)
            u_list.append(u)
            v_list.append(v)

        enhanced = torch.clamp(y / (x_t + 1e-6), 0.0, 1.0)

        # For Ultralytics model chaining we must return a Tensor (the enhanced image).
        # If downstream code needs the intermediate lists for debugging, consider
        # exposing them via an attribute or a separate debug-only method.
        return enhanced


class MobiVari(nn.Module):
    """MobiVari: Modified MobileNet V2 variant for D-RAMiT module.
    
    Modifications from original MobileNet V2:
    - ReLU6 replaced with LeakyReLU activation
    - First convolution replaced with group convolution
    - Conditional residual connections (omitted when channel mismatch)
    """
    
    def __init__(self, c1, c2, expansion=6, groups=1):
        super().__init__()
        c_hidden = c1 * expansion
        
        # Group convolution instead of standard expansion conv
        self.conv1 = nn.Conv2d(c1, c_hidden, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(c_hidden)
        
        # Depthwise convolution
        self.conv2 = nn.Conv2d(c_hidden, c_hidden, 3, padding=1, groups=c_hidden, bias=False)
        self.bn2 = nn.BatchNorm2d(c_hidden)
        
        # Pointwise convolution
        self.conv3 = nn.Conv2d(c_hidden, c2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        
        # LeakyReLU instead of ReLU6
        self.act = nn.LeakyReLU(0.1)
        
        # Residual connection only when input/output channels match
        self.use_residual = (c1 == c2)
        
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Conditional residual connection
        if self.use_residual:
            out = out + x
            
        return out


class SpatialSelfAttention(nn.Module):
    """Spatial Self-Attention (SPSA) for capturing fine-grained local information."""
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        q = self.query(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.key(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.value(x).view(B, self.num_heads, self.head_dim, H * W)
        
        # Compute attention weights
        attn = (q.transpose(-2, -1) @ k) * self.scale  # [B, heads, HW, HW]
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  # [B, heads, head_dim, HW]
        out = out.contiguous().view(B, C, H, W)
        
        return self.proj(out)


class ChannelSelfAttention(nn.Module):
    """Channel Self-Attention (CHSA) for modeling inter-channel dependencies."""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        
        self.scale = channels ** -0.5
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Global average pooling to get channel-wise features
        x_gap = F.adaptive_avg_pool2d(x, 1).view(B, C)  # [B, C]
        
        # Generate Q, K, V
        q = self.query(x_gap)  # [B, C]
        k = self.key(x_gap)    # [B, C]
        v = self.value(x_gap)  # [B, C]
        
        # Compute attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, C, C]
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = attn @ v  # [B, C]
        out = self.proj(out)
        
        # Broadcast back to spatial dimensions
        out = out.view(B, C, 1, 1).expand(B, C, H, W)
        
        return out


class DRAMiT(nn.Module):
    """Dimension Reciprocal Attention Module (D-RAMiT) from DarkYOLO.
    
    Enhances model's ability to capture both global information and local details
    through parallel Spatial Self-Attention (SPSA) and Channel Self-Attention (CHSA).
    Includes MobiVari for feature transformation and FFN with LayerNorm.
    """
    
    def __init__(self, c1, c2, num_heads=8, ffn_expansion=4):
        super().__init__()
        # If c1 != c2, we need a projection layer
        if c1 != c2:
            self.proj = nn.Conv2d(c1, c2, 1, bias=False)
        else:
            self.proj = None
            
        self.channels = c2
        
        # Head Split - divide channels for parallel attention
        self.head_split_channels = c2 // 2
        
        # Spatial Self-Attention (SPSA)
        self.spsa = SpatialSelfAttention(self.head_split_channels, num_heads//2)
        
        # Channel Self-Attention (CHSA)
        self.chsa = ChannelSelfAttention(self.head_split_channels)
        
        # MobiVari module for feature transformation
        self.mobivari = MobiVari(c2, c2, expansion=6, groups=4)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        
        # Feed-Forward Network (FFN)
        ffn_hidden = c2 * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(c2, ffn_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(ffn_hidden, c2)
        )
        
    def forward(self, x):
        # Apply projection if needed
        if self.proj is not None:
            x = self.proj(x)
            
        residual = x
        B, C, H, W = x.shape
        
        # Head Split operation
        x1, x2 = torch.chunk(x, 2, dim=1)  # Split channels
        
        # Parallel attention mechanisms
        spsa_out = self.spsa(x1)  # Spatial Self-Attention
        chsa_out = self.chsa(x2)  # Channel Self-Attention
        
        # Concatenate attention outputs (Equation 11)
        attention_concat = torch.cat([spsa_out, chsa_out], dim=1)
        
        # MobiVari feature transformation
        mobivari_out = self.mobivari(attention_concat)
        
        # First LayerNorm (need to reshape for LayerNorm)
        x_norm = mobivari_out.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
        x_norm = self.norm1(x_norm)
        
        # Feed-Forward Network
        ffn_out = self.ffn(x_norm)
        
        # Second LayerNorm
        ffn_out = self.norm2(ffn_out)
        
        # Reshape back to spatial dimensions
        ffn_out = ffn_out.transpose(1, 2).view(B, C, H, W)
        
        # Skip connection (residual)
        return ffn_out + residual


class CSPPF(nn.Module):
    """Cross-Spatial Pyramid Pooling Feature (CSPPF) module from DarkYOLO.
    
    Implementation based on the diagram in Figure 3, which shows the actual architecture
    used in the paper (equations and text descriptions have inconsistencies).
    
    Architecture follows the visual diagram with multiple concatenation points
    and interconnected pooling operations for enhanced low-light feature extraction.
    """
    
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        
        # If c1 != c2, we need a projection layer
        if c1 != c2:
            self.proj = nn.Conv2d(c1, c2, 1, bias=False)
        else:
            self.proj = None
        
        # Initial convolution as shown in diagram
        self.cv1 = nn.Conv2d(c2, c_, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        
        # Pooling operations (using same kernel size for all)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
        self.avgpool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k//2)
        
        # Final convolution - input channels based on diagram concatenations
        # From diagram: original x + 6 pooling outputs = 7 feature maps
        self.cv2 = nn.Conv2d(7 * c_, c2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        
    def forward(self, x):
        # Apply projection if needed
        if self.proj is not None:
            x = self.proj(x)
        
        # Initial convolution
        x = self.act(self.bn1(self.cv1(x)))
        
        # Following the diagram architecture:
        # Upper branch: Max -> Avg -> Max
        upper_max1 = self.maxpool(x)
        upper_avg = self.avgpool(upper_max1)  
        upper_max2 = self.maxpool(upper_avg)
        
        # Lower branch: Avg -> Max -> Avg  
        lower_avg1 = self.avgpool(x)
        lower_max = self.maxpool(lower_avg1)
        lower_avg2 = self.avgpool(lower_max)
        
        # Concatenate all features as shown in diagram
        # Original x + all 6 pooling outputs
        y = torch.cat([
            x,              # Original features
            upper_max1,     # First max from upper branch
            upper_avg,      # Avg from upper branch
            upper_max2,     # Final max from upper branch
            lower_avg1,     # First avg from lower branch
            lower_max,      # Max from lower branch  
            lower_avg2      # Final avg from lower branch
        ], dim=1)
        
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
