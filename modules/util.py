from torch import nn

import torch.nn.functional as F
import torch

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d

import numpy as np
from paddle import fluid
from paddle.fluid import dygraph
def kp2gaussian(kp, spatial_size, kp_variance: np.ndarray) -> np.ndarray:
    """
    Transform a keypoint into gaussian like representation
    BP is not supported
    """
    if isinstance(kp, fluid.core_avx.VarBase):
        mean = kp.numpy()
    elif isinstance(kp, np.ndarray):
        mean = kp
    else:
        raise TypeError('TYPE of keypoint : %s is not supported'%type(kp))

    coordinate_grid = make_coordinate_grid_cpu(spatial_size, mean.type)
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.reshape(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.reshape(*shape)

    mean_sub = (coordinate_grid - mean)

    out = np.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

# Type is Always Float32
make_coordinate_grid_cpu = lambda spatial_size, ttype: np.stack(np.meshgrid(np.linspace(-1, 1, spatial_size[1]), np.linspace(-1, 1, spatial_size[0])), axis=-1).astype(np.float32)
make_coordinate_grid = lambda spatial_size, ttype: dygraph.to_variable(make_coordinate_grid_cpu(spatial_size, ttype))

######################################################################
# def make_coordinate_grid(spatial_size, type):
#     """
#     Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
#     """
#     h, w = spatial_size
#     x = torch.arange(w).type(type)
#     y = torch.arange(h).type(type)
#
#     x = (2 * (x / (w - 1)) - 1)
#     y = (2 * (y / (h - 1)) - 1)
#
#     yy = y.view(-1, 1).repeat(1, w)
#     xx = x.view(1, -1).repeat(h, 1)
#
#     meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
######################################################################

class ResBlock2d(dygraph.Layer):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding, **kwargs):
        super(ResBlock2d, self).__init__(**kwargs)
        Conv2D = lambda in_channels, out_channels, kernel_size, padding: dygraph.Conv2D(num_channels=in_channels, num_filters=out_channels, filter_size=kernel_size, padding=padding)
        self.conv1 = Conv2D(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = Conv2D(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        # TODO: rewrite new BN
        self.norm1 = dygraph.BatchNorm(num_channels=in_features)
        self.norm2 = dygraph.BatchNorm(num_channels=in_features)

    def forward(self, x):
        out = self.norm1(x)
        out = fluid.layers.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = fluid.layers.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(dygraph.Layer):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()
        Conv2D = lambda in_channels, out_channels, kernel_size, padding, groups: dygraph.Conv2D(num_channels=in_channels,
                                                                                        num_filters=out_channels,
                                                                                        filter_size=kernel_size,
                                                                                        padding=padding,
                                                                                        groups=groups)
        self.conv = Conv2D(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = dygraph.BatchNorm(num_channels=out_features)

    def forward(self, x):
        out = fluid.layers.interpolate(x, scale=2, resample='NEAREST')
        out = self.conv(out)
        out = self.norm(out)
        out = fluid.layers.relu(out)
        return out


class DownBlock2d(dygraph.Layer):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        Conv2D = lambda in_channels, out_channels, kernel_size, padding, groups: dygraph.Conv2D(num_channels=in_channels, num_filters=out_channels, filter_size=kernel_size, padding=padding, groups=groups)
        self.conv = Conv2D(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = dygraph.BatchNorm(out_features)
        self.pool = dygraph.Pool2D(pool_size=(2, 2), pool_type='avg', pool_stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = fluid.layers.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(dygraph.Layer):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        Conv2D = lambda in_channels, out_channels, kernel_size, padding, groups: dygraph.Conv2D(
            num_channels=in_channels, num_filters=out_channels, filter_size=kernel_size, padding=padding, groups=groups)
        self.conv = Conv2D(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = dygraph.BatchNorm(out_features)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = fluid.layers.relu(out)
        return out


class Encoder(dygraph.Layer):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = dygraph.LayerList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(dygraph.Layer):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = dygraph.LayerList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            # TODO: If the size of width or length is odd, out and skip cannot concat
            out = fluid.layers.concat([out, skip], axis=1)
        return out


class Hourglass(dygraph.Layer):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))

# TODO: 20200810
class AntiAliasInterpolation2d(dygraph.Layer):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        # TODO: kernel DO NOT NEED BP, initialized in cpu by numpy
        meshgrids = np.meshgrid(
            [
                np.arange(size, dtype=np.float32)
                for size in kernel_size
                ]
        )
        meshgrids = [i.T for i in meshgrids]
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= np.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.reshape(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        return out
