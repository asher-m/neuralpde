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

class GaussianDistanceWeight(nn.Module):
    def __init__(
            self,
            *coordinates: Tuple[torch.Tensor]
        ):
        """
        Compute the Gaussian weight across an array of coordinates to a point x.  Outputs
        an array of weights.

        Args:
            coordinates:        Tuple of 1-D tensors of coordinate ranges.  Coordinate ranges are usually
                                monotonic, but I don't check and I don't think it will break anything.
        
        Every tuple in *coordinates should have the same number of coordinate ranges (i.e., the same spatial dimension,)
        and corresponding coordinate ranges should have the same length.  For example, if coordinates = ((x1, y1), (x2, y2)),
        then len(x1) = len(x2) and len(y1) = len(y2).
        """
        super().__init__()

        if len(coordinates) == 1:
            self.coordinates = nn.Buffer(
                torch.stack(torch.meshgrid(*coordinates[0], indexing='xy'))
            )
            self.batched = False
        elif len(coordinates) > 1:
            self.coordinates = nn.Buffer(
                torch.stack([torch.stack(torch.meshgrid(*c, indexing='xy')) for c in coordinates])
            )
            self.batched = True
        else:
            raise ValueError('Received no coordinate ranges!')

        self.dims = len(coordinates[0])
        """ Number of spatial dimensions. """

        self.height = nn.Parameter(torch.ones(1))
        self.width = nn.Parameter(torch.ones(1))

    def forward(
            self,
            x: torch.tensor
        ):
        """
        Args:
            x:      A 2-D tensor where first dimension corresponds to batch, and second dimension
                    is a collection of coordinates.

        `x.shape[1]` must be equal to the number of ranges in each (inner) tuple in `coordinates` passed to this
        layer and initialization, (i.e., we must have received a position in each axis.)
        """
        if len(x.shape) < 1 or len(x.shape) > 2:  # received in mistake
            raise ValueError('Received an misshapen array of x values!')

        elif self.batched and len(x.shape) == 2:  # batched coords, batched x
            if not x.shape[1] == self.dims:
                raise ValueError('Received incompatible x with number of coordinates from initialization!')
            d = torch.sqrt(torch.sum((self.coordinates - x[:, :, *(None,) * self.dims])**2, 1))

        elif not self.batched and len(x.shape) == 2:  # unbatched coords, batched x
            if not x.shape[1] == self.dims:
                raise ValueError('Received incompatible x with number of coordinates from initialization!')
            d = torch.sqrt(torch.sum((self.coordinates[None, ...] - x[:, :, *(None,) * self.dims])**2, 1))

        elif self.batched and len(x.shape) == 1:  # batched but want weights for one x
            if not x.shape[0] == self.dims:
                raise ValueError('Received incompatible x with number of coordinates from initialization!')
            d = torch.sqrt(torch.sum((self.coordinates - x[None, :, *(None,) * self.dims])**2, 1))

        elif not self.batched and len(x.shape) == 1:  # unbatched and want weights for one x
            if not x.shape[0] == self.dims:
                raise ValueError('Received incompatible x with number of coordinates from initialization!')
            d = torch.sqrt(torch.sum((self.coordinates - x[:, *(None,) * self.dims])**2, 0))

        else:
            raise ValueError('Failed conditions above!  This should not be possible!')

        return self.height * torch.exp(-1/2 * d / self.width)
