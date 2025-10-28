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
        gamma_scale = self.fc(x)  # [B, C]
        # use the batch-agnostic base gamma multiplied by (0.5 + scale)
        gamma = torch.clamp(self.base_gamma * (0.5 + gamma_scale.mean(0)), 0.5, 2.0)
        out = torch.pow(x + 1e-6, gamma.view(1, -1, 1, 1))
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
        x = torch.clamp(x, 0, 1)
        mean = F.adaptive_avg_pool2d(x, self.grid_size)
        std = torch.sqrt(
            F.adaptive_avg_pool2d((x - F.interpolate(mean, (H, W))) ** 2, self.grid_size) + 1e-6
        )
        mean_up = F.interpolate(mean, (H, W), mode="bilinear", align_corners=False)
        std_up = F.interpolate(std, (H, W), mode="bilinear", align_corners=False)
        enhanced = (x - mean_up) / (std_up + 1e-6)
        enhanced = enhanced * self.alpha.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        enhanced = torch.clamp(enhanced * 0.5 + 0.5, 0, 1)
        return enhanced


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
        out = []
        # Apply channel-wise blur + retinex
        for c in range(C):
            I = x[:, c : c + 1, :, :]
            blur = F.conv2d(I, self.gaussian_kernel, padding=self.kernel_size // 2)
            retinex = torch.log(I + self.eps) - torch.log(blur + self.eps)
            enhanced = self.alpha[c] * retinex
            out.append(enhanced)
        out = torch.cat(out, dim=1)
        return torch.clamp(out, -2, 2)
