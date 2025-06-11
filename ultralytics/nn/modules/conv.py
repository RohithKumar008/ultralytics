# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
import torch.nn.functional as F
__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
    "DWSConv",
    "CondConv",
    "SE",
    "DeformableConv",
    "SAMO",
    "ECA",
    "SimpleGate",
    "MobileViT",
    "DenseBlock",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        out = self.act(self.bn(self.conv(x)))
        # print("Output of Conv(BN):", out.shape)
        return out

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        # print("Output of Conv:", x.shape)
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """
        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2  # mid channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)

        # Only use groups=c_ if c_ divides c_
        cheap_groups = c_ if c_ == 1 or c_ == c_ else 1
        self.cv2 = Conv(c_, c_, 5, 1, None, cheap_groups, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

class CBAM(nn.Module):
    def __init__(self, c1, c2=None):  # ✅ Add c2 for compatibility
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: List[torch.Tensor]):
        """
        Select and return a particular index from input.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]


class DWSConv(nn.Module):
    """
    Depthwise Separable Convolution with BatchNorm and Activation.
    
    Attributes:
        depthwise (nn.Conv2d): Depthwise convolution layer.
        pointwise (nn.Conv2d): Pointwise convolution layer.
        bn (nn.BatchNorm2d): Batch normalization.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation (SiLU).
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, d=1, act=True):
        """
        Initialize the DWSConv layer.
        
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding. Defaults to autopad.
            d (int): Dilation.
            act (bool | nn.Module): Activation type.
        """
        super().__init__()
        self.depthwise = nn.Conv2d(c1, c1, k, s, autopad(k, p, d), groups=c1, dilation=d, bias=False)
        self.pointwise = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Forward pass applying depthwise → pointwise → BN → activation.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        out = self.act(self.bn(x))
        print("Output of DWSConv:", out.shape)
        return out


class CondConv(nn.Module):
    """
    Conditional Convolution Layer with multiple expert kernels and input-dependent routing.

    Attributes:
        expert_weights (nn.Parameter): Learnable expert kernels.
        expert_bias (nn.Parameter): Learnable expert biases.
        routing_fn (nn.Module): Learns expert weights based on input.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, d=1, act=True, num_experts=4):
        """
        Initialize CondConv layer.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding. Uses autopad if None.
            d (int): Dilation.
            act (bool | nn.Module): Activation.
            num_experts (int): Number of expert convolution kernels.
        """
        super().__init__()
        self.k = k
        self.s = s
        self.d = d
        self.p = autopad(k, p, d)
        self.num_experts = num_experts

        # Expert weights and bias
        self.expert_weights = nn.Parameter(torch.randn(num_experts, c2, c1, k, k))
        self.expert_bias = nn.Parameter(torch.randn(num_experts, c2))

        # Routing function (global avg pooling + FC)
        self.routing_fn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, num_experts),
            nn.Sigmoid()
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        B = x.size(0)
        routing_weights = self.routing_fn(x)  # (B, num_experts)

        # Weighted combination of expert weights and biases
        weight = torch.einsum('be,eocij->bocij', routing_weights, self.expert_weights)  # (B, c2, c1, k, k)
        bias = torch.einsum('be,eo->bo', routing_weights, self.expert_bias)             # (B, c2)

        # Apply sample-wise convolution
        outputs = []
        for i in range(B):
            xi = x[i:i+1]
            wi = weight[i]
            bi = bias[i]
            yi = F.conv2d(xi, wi, bi, stride=self.s, padding=self.p, dilation=self.d)
            outputs.append(yi)

        out = torch.cat(outputs, dim=0)  # (B, c2, H_out, W_out)
        out = self.act(self.bn(out))
        print("Output of condconv:", out.shape)
        return out

class SE(nn.Module):
    """
    Squeeze-and-Excitation (SE) block using Conv2d for channel recalibration.
    """

    def __init__(self, c1, reduction=16):
        super().__init__()
        c_ = max(1, c1 // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_, c1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = x * self.fc(self.pool(x))
        print("SE output : ",out.shape)
        return out


class DeformableConv(nn.Module):
    """
    Deformable Convolution block with offset learning and activation.

    Args:
        in_ch (int): Input channels.
        out_ch (int): Output channels.
        k (int): Kernel size.
        s (int): Stride.
    """

    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_ch, 2 * k * k, kernel_size=k, stride=s, padding=k // 2)
        self.deform_conv = DeformConv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        out = self.act(self.bn(x))
        print("deformable conv shape :",out.shape)
        return out
        
class SAMO(nn.Module):
    """
    Small Attention Modulation Operator (SAMO).

    Applies a spatial attention mask to emphasize regions likely to contain small objects.

    Attributes:
        conv (nn.Conv2d): 1×1 convolution to predict attention scores.
        sigmoid (nn.Sigmoid): Sigmoid activation for attention mask.
    """

    def __init__(self, c1, c2=None):
        """
        Initialize SAMO layer.

        Args:
            c1 (int): Number of input channels.
            c2 (int, optional): Ignored. Included for compatibility.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of SAMO.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Modulated feature map.
        """
        mask = self.sigmoid(self.conv(x))  # [B, 1, H, W]
        return x * mask  # Broadcasted element-wise modulation

import torch
import torch.nn as nn

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    """

    def __init__(self, c1, c2=None, k_size=3):  # c2 is ignored, kept for compatibility
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)             # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(1, 2)  # [B, 1, C]
        y = self.conv(y)                   # [B, 1, C]
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)
        
class SimpleGate(nn.Module):
    """
    SimpleGate module.

    Splits input along channel dimension and performs element-wise multiplication.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=False):
        """
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool): Activation (default=False).
        """
        super().__init__()
        assert c2 % 2 == 0, "SimpleGate output channels must be even"
        self.conv = Conv(c1, c2, k=k, s=s, p=p, g=g, d=d, act=act)

    def forward(self, x):
        x = self.conv(x)
        c = x.shape[1] // 2
        out = x[:, :c] * x[:, c:]
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileViT(nn.Module):
    def __init__(self, in_channels=512, transformer_dim=192, patch_size=(2, 2), depth=2, num_heads=4):
        super(MobileViT, self).__init__()
        self.patch_size = patch_size

        # 1. Local representation (depthwise + pointwise)
        self.local_rep = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # depthwise
            nn.Conv2d(in_channels, transformer_dim, kernel_size=1),  # pointwise
        )

        # 2. Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, batch_first=True),
            num_layers=depth
        )

        # 3. Fusion projection
        self.fusion = nn.Sequential(
            nn.Conv2d(transformer_dim, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        print("Input to MobileViT:", x.shape)  # Expecting [B, 512, 4, 4]

        B, C, H, W = x.shape
        ph= 2
        pw =2

        # Step 1: Local representation
        x = self.local_rep(x)  # [B, transformer_dim, 4, 4]
        print("After local rep:", x.shape)

        # Step 2: Flatten into patches
        x = x.reshape(B, -1, ph, H // ph, pw, W // pw)  # [B, C, 2, 2, 2, 2]
        x = x.permute(0, 3, 5, 2, 4, 1)  # [B, H//ph, W//pw, ph, pw, C]
        x = x.reshape(B * (H // ph) * (W // pw), ph * pw, -1)  # [B*4, 4, transformer_dim]

        # Step 3: Transformer
        x = self.transformer(x)  # [B*4, 4, transformer_dim]

        # Step 4: Reshape back to image
        x = x.reshape(B, H // ph, W // pw, ph, pw, -1).permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, -1, H, W)  # [B, transformer_dim, 4, 4]

        # Step 5: Fusion
        x = self.fusion(x)  # [B, in_channels, 4, 4]
        return x
        
class DenseBlock(nn.Module):
    """
    DenseBlock using DWSConv as the basic layer.
    Arguments:
        num_layers: Number of layers inside the block
        in_channels: Input channels
        growth_rate: Channels to add per layer
    Output:
        Concatenated features from all layers
    """
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(
                DWSConv(channels, growth_rate, kernel_size=3, stride=1, padding=1)
            )
            channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
            out = torch.cat(features, dim=1)
            print("dense layer output : ",out.shape)
        return out
        
globals()['DWSConv'] = DWSConv
globals()['CondConv'] = CondConv
globals()['SE'] = SE
globals()['DeformableConv'] = DeformableConv
globals()['DenseBlock'] = DenseBlock
globals()['SAMO'] = SAMO
globals()['ECA'] = ECA
globals()['SimpleGate'] = SimpleGate
globals()['MobileViT'] = MobileViT
