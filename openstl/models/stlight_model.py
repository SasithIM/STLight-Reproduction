import torch
from torch import nn


class STLight_Model(nn.Module):
    r"""STLight Model

    Implementation of `STLight: a Fully Convolutional Approach for Efficient
    Predictive Learning by Spatio-Temporal joint Processing
    <https://arxiv.org/abs/2411.10198>`_.

    Faithful reproduction of the official implementation released by the
    authors in Appendix G (Figure 10) of the arXiv version, combined with the
    Kaiming-Normal initialization of Appendix D (Listing 1). Following the
    paper's "Hybrid" initialization strategy, the patch reassemble layer keeps
    PyTorch's default (uniform Kaiming) initialization.
    """

    def __init__(self, in_shape, aft_seq_length, dim=224, depth=16,
                 patch_size=2, overlapping=0, K_1=3, K_2=7, **kwargs):
        super(STLight_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        self.in_layers = T * C
        self.out_layers = aft_seq_length * C
        self.patch_encoder = PatchEncoder(self.in_layers, dim, patch_size,
                                          overlapping)
        self.net = nn.ModuleList([STLMixer(dim, K_1, K_2) for _ in range(depth)])
        self.patch_reassemble = nn.Conv2d(dim // patch_size**2,
                                          self.out_layers, kernel_size=1)
        self.up = nn.PixelShuffle(patch_size)
        self.patch_encoder.apply(self._init_weights)
        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        # Appendix D, Listing 1: Kaiming-Normal initialization.
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B, T * C, H, W)
        x = self.patch_encoder(x)

        for i, block in enumerate(self.net):
            if i == len(self.net) // 3:
                x1 = x  # skip connection from the block at depth // 3
            if i == 2 * len(self.net) // 3:
                x = x + x1
            x = block(x)

        x = self.up(x)
        x = self.patch_reassemble(x)
        return x.reshape(B, self.out_layers // C, C, H, W)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def STLMixer(dim, K_1, K_2):
    return nn.Sequential(
        Residual(  # depthwise convolution block
            nn.Sequential(
                nn.Conv2d(dim, dim, K_1, groups=dim, padding='same'),
                nn.Conv2d(dim, dim, K_2, groups=dim, padding='same',
                          dilation=3),
                nn.GELU(),
                nn.BatchNorm2d(dim),
            )
        ),
        nn.Sequential(  # pointwise convolution block
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(), nn.BatchNorm2d(dim),
        ),
    )


def PatchEncoder(in_layers, dim, patch_size, overlapping):
    return nn.Sequential(
        nn.Conv2d(in_layers, dim,
                  kernel_size=patch_size * max(1, overlapping),
                  stride=patch_size,
                  padding=max(0, overlapping - 1) * patch_size // 2),
        nn.BatchNorm2d(dim),
        nn.GELU(),
    )
