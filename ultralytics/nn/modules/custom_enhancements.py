"""Custom enhancement modules for Ultralytics models.

This file contains a drop-in, YAML-friendly version of AdaptivePerChannelGamma
adapted to the model parser expectations (i.e. constructors accept c1, c2, ...).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePerChannelGamma(nn.Module):
    """Adaptive per-channel gamma correction compatible with parse_model.

    The parser will call AdaptivePerChannelGamma(c1, c2, *args) where c1 is the
    input channels inferred from the "from" field and c2 is the YAML args[0]
    (desired output channels). This class accepts that calling convention and
    uses the channel count to build the internal FC used to generate per-
    channel gamma scales.

    Args:
        c1 (int): input channels inferred by the parser (ignored for internal
            sizing but kept for compatibility).
        c2 (int | None): output channels (preferred channel count for internal
            parameters). If None, c1 is used.
        num_channels (int | None): explicit number of channels to use (overrides
            c1/c2). Mostly useful if you want to hardcode a different channel size.
    """

    def __init__(self, c1: int, c2: int = None, num_channels: int = None):
        super().__init__()
        ch = num_channels if num_channels is not None else (c2 if c2 is not None else c1)

        # small MLP that computes a per-channel scale in (0,1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch),
            nn.Sigmoid(),
        )

        # base gamma per channel (learnable)
        self.base_gamma = nn.Parameter(torch.ones(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        # Clamp input to prevent extreme values
        x = torch.clamp(x, -10.0, 10.0)
        
        gamma_scale = self.fc(x)  # [B, C]
        # use the batch-agnostic base gamma multiplied by (0.5 + scale)
        gamma = torch.clamp(self.base_gamma * (0.5 + gamma_scale.mean(0)), 0.8, 1.2)  # More conservative range
        
        # Ensure input is positive for gamma correction
        x_positive = torch.clamp(x, 1e-8, 100.0)  # Clamp to positive range
        out = torch.pow(x_positive, gamma.view(1, -1, 1, 1))
        
        # Clamp output to prevent extreme values
        out = torch.clamp(out, -10.0, 10.0)
        return out


class LearnableCLAHE(nn.Module):
    """Differentiable, learnable CLAHE-like module compatible with YAML parser.

    Constructor signature mirrors parser expectations: (c1, c2=None, grid_size=(8,8)).
    """

    def __init__(self, c1: int, c2: int = None, grid_size=(8, 8)):
        super().__init__()
        ch = c2 if c2 is not None else c1
        self.grid_size = grid_size
        self.alpha = nn.Parameter(torch.ones(ch))
        self.beta = nn.Parameter(torch.zeros(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Don't normalize input - work with original feature maps
        x = torch.clamp(x, -10.0, 10.0)  # Prevent extreme values
        
        mean = F.adaptive_avg_pool2d(x, self.grid_size)
        mean_up = F.interpolate(mean, (H, W), mode="bilinear", align_corners=False)
        
        # Calculate variance more stably
        diff = x - mean_up
        var = F.adaptive_avg_pool2d(diff ** 2, self.grid_size)
        std = torch.sqrt(var + 1e-8)  # Larger epsilon for stability
        std_up = F.interpolate(std, (H, W), mode="bilinear", align_corners=False)
        
        # Normalize with clamped parameters
        alpha_clamped = torch.clamp(self.alpha, 0.1, 2.0)
        beta_clamped = torch.clamp(self.beta, -1.0, 1.0)
        
        enhanced = (diff) / (std_up + 1e-6)
        enhanced = enhanced * alpha_clamped.view(1, -1, 1, 1) + beta_clamped.view(1, -1, 1, 1)
        
        # Add residual connection for stability
        enhanced = x + 0.1 * enhanced  # Small enhancement factor
        
        return torch.clamp(enhanced, -10.0, 10.0)


class TrainableRetinex(nn.Module):
    """Trainable Retinex-style enhancement compatible with YAML parser.

    Signature: (c1, c2=None, kernel_size=15, sigma=5.0)
    """

    def __init__(self, c1: int, c2: int = None, kernel_size: int = 15, sigma: float = 5.0):
        super().__init__()
        ch = c2 if c2 is not None else c1
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.eps = 1e-6

        # learnable per-channel scale
        self.alpha = nn.Parameter(torch.ones(ch))

        # fixed Gaussian kernel buffer
        self.register_buffer("gaussian_kernel", self._create_gaussian_kernel(kernel_size, sigma))

    def _create_gaussian_kernel(self, ksize, sigma):
        ax = torch.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, ksize, ksize)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Clamp input to prevent extreme values
        x = torch.clamp(x, -10.0, 10.0)
        
        out = []
        # Apply channel-wise blur + retinex
        for c in range(C):
            I = x[:, c : c + 1, :, :]
            
            # Make input positive for log operations
            I_positive = torch.abs(I) + self.eps
            
            blur = F.conv2d(I_positive, self.gaussian_kernel, padding=self.kernel_size // 2)
            blur = torch.clamp(blur, self.eps, 100.0)  # Prevent extreme blur values
            
            # Stable log operations
            log_I = torch.log(I_positive + self.eps)
            log_blur = torch.log(blur + self.eps)
            
            retinex = log_I - log_blur
            retinex = torch.clamp(retinex, -5.0, 5.0)  # Prevent extreme retinex values
            
            # Clamp alpha to prevent instability
            alpha_clamped = torch.clamp(self.alpha[c], 0.1, 1.0)
            enhanced = alpha_clamped * retinex
            
            out.append(enhanced)
            
        out = torch.cat(out, dim=1)
        
        # Add residual connection for stability
        out = x + 0.1 * out  # Small enhancement factor
        
        return torch.clamp(out, -10.0, 10.0)
