import torch
import torch.nn as nn

from functools import reduce
from typing import Tuple



class LocallyConnected2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            in_spatial_shape: Tuple[int],
            kernel_size: int,
            dilation: int = 1,
            padding: int = 0,
            stride: int = 1,
        ):
        """
        Initializes a locally-connected 2d layer.

        This is a simple implementation that I can expand later if I want or need.

        See [the PyTorch docs on `torch.nn.Unfold`](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html)
        for a more thorough description of these parameters and how they work.  In particular, I've tried
        to reuse the notational convention described in that documentation here.

        Note: Currently, only 4-D input tensors (batched image-like tensors) are supported,
        as per `nn.Unfold` and `nn.functional.unfold`.

        Args:
            in_channels         : Number of channels in the input image (e.g., 3 for RGB).
            out_channels        : Number of output channels (number of filters per spatial location).
            in_spatial_shape    : Input spatial shape (spatial_1 x spatial_2 x ...).
            kernel_size         : The size of the sliding window.
            dilation            : Stride of elements within a kernel neighborhood.
            stride              : Stride for the sliding window.
            padding             : Zero-padding added to both sides of the input.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_spatial_shape = in_spatial_shape
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        self.out_spatial_size = tuple(
            (
                (self.in_spatial_shape[d] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
                for d in range(len(self.in_spatial_shape))
            )
        )
        """ Output spatial shape """

        # Total number of blocks from unfold (unused, but also helped me write this in the first place)
        # I want to keep this here to prevent this from descending into the arcane
        # I.e., the process is important!
        self.L = reduce(lambda x, y: x * y, self.out_spatial_size)
        """ Total number of blocks """

        self._block_size = self.in_channels * self.kernel_size**len(self.in_spatial_shape)
        """ Number of elements in each block """

        self.weight = nn.Parameter(torch.ones(out_channels, *self.out_spatial_size, self._block_size) / self._block_size)
        self.bias = nn.Parameter(torch.randn(out_channels, *self.out_spatial_size))

    def forward(self, x: torch.tensor):
        """
        Note: Currently, only 4-D input tensors (batched image-like tensors) are supported,
        as per `nn.Unfold` and `nn.functional.unfold`.  This method will add a degenerate axis
        at position 0 if it receives a tensor of ndim = 3, otherwise it will throw an error.

        Args:
            x (Tensor): Input tensor with shape (batch_size, in_channels, *self.in_spatial_shape)

        Returns:
            out (Tensor): Output tensor with shape (batch_size, out_channels, *self.out_spatial_shape)
        """
        if x.ndim == 4:
            batch_size = x.size(0)

            blocks = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
            out = torch.einsum('bil,oli->bol', blocks, self.weight.flatten(1, -2)) + self.bias.flatten(1, -1).unsqueeze(0)
            out = out.reshape((batch_size, self.out_channels, *self.out_spatial_size))
            return out

        elif x.ndim == 3:
            x = x.unsqueeze(0)
            batch_size = 1

            blocks = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
            out = torch.einsum('bil,oli->bol', blocks, self.weight.flatten(1, -2)) + self.bias.flatten(1, -1).unsqueeze(0)
            out = out.reshape((batch_size, self.out_channels, *self.out_spatial_size))
            return out.squeeze(0)

        else:
            raise NotImplementedError(
                'Only 4-D (batched image-like tensors) or 3-D tensors (unbatched image-like tensors) are supported!\n'
                'nn.Unfold and nn.functional.unfold only support 4-D tensors.'
            )
