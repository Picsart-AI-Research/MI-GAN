import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class lrelu_agc:
    """
    The lrelu layer with alpha, gain and clamp
    """

    def __init__(self, alpha=0.2, gain=1, clamp=None):
        self.alpha = alpha
        if gain == 'sqrt_2':
            self.gain = np.sqrt(2)
        else:
            self.gain = gain
        self.clamp = clamp

    def __call__(self, x, gain=1):
        x = F.leaky_relu(x, negative_slope=self.alpha, inplace=True)
        act_gain = self.gain * gain
        act_clamp = self.clamp * gain if self.clamp is not None else None
        if act_gain != 1:
            x = x * act_gain
        if act_clamp is not None:
            x = x.clamp(-act_clamp, act_clamp)
        return x


def setup_filter(f, device=torch.device('cpu'), normalize=True, flip_filter=False, gain=1, separable=None):
    # Validate.
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    # Separable?
    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f


class Downsample2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=4,
            groups=in_channels,
            padding=1,
            bias=False,
            stride=2
        )
        f = setup_filter([1, 3, 3, 1], gain=1)
        self.filter.weight = nn.Parameter(f.repeat([*self.filter.weight.shape[:2], 1, 1]))

    def forward(self, x):
        x = self.filter(x)
        return x


class Upsample2d(nn.Module):
    def __init__(self, in_channels, resolution=None):
        super().__init__()
        self.nearest_up = nn.Upsample(scale_factor=2, mode='nearest')
        w = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float32)
        assert resolution is not None
        self.register_buffer('filter_const', w.repeat(1, 1, resolution//2, resolution//2))

        self.filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=4,
            groups=in_channels,
            bias=False
        )

        f = setup_filter([1, 3, 3, 1], gain=4)
        self.filter.weight = nn.Parameter(f.repeat([*self.filter.weight.shape[:2], 1, 1]))

    def forward(self, x):
        x = self.nearest_up(x)
        x = x * self.filter_const
        x = F.pad(x, pad=(2, 1, 2, 1))
        x = self.filter(x)
        return x


class SeparableConv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            bias=True,
            activation=None,
            resolution=None,
            use_noise=False,
            down=1,
            up=1
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
            groups=in_channels
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            groups=1
        )

        self.downsample = None
        if down > 1:
            self.downsample = Downsample2d(in_channels)

        self.upsample = None
        if up > 1:
            self.upsample = Upsample2d(out_channels, resolution=resolution)

        self.use_noise = use_noise
        if use_noise:
            assert resolution is not None
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        if self.activation is not None:
            x = self.activation(x)

        if self.downsample is not None:
            x = self.downsample(x)
        x = self.conv2(x)
        if self.upsample is not None:
            x = self.upsample(x)

        if self.use_noise:
            noise = self.noise_const * self.noise_strength
            x = x.add_(noise)
        if self.activation is not None:
            x = self.activation(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        ic_n,
        oc_n,
        rgb_n=None,
        activation=lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256),
        down=2
    ):
        super().__init__()

        self.fromrgb = None
        if rgb_n is not None:
            self.fromrgb = nn.Conv2d(rgb_n, ic_n, 1)

        self.conv1 = SeparableConv2d(ic_n, ic_n, 3, activation=activation)
        self.conv2 = SeparableConv2d(ic_n, oc_n, 3, activation=activation, down=down)
        self.activation = activation

    def forward(self, x, img):
        if self.fromrgb is not None:
            y = self.fromrgb(img)
            y = self.activation(y)
            x = x + y if x is not None else y

        feat = self.conv1(x)
        x = self.conv2(feat)
        return x, feat


class Encoder(nn.Module):
    def __init__(
        self,
        resolution=256,
        ic_n=4,
        ch_base=32768,
        ch_max=512,
        activation=lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256)
    ):
        super().__init__()

        log2res = int(np.log2(resolution))
        if 2 ** log2res != resolution:
            raise ValueError
        self.encode_res = [2 ** i for i in range(log2res, 1, -1)]
        self.ic_n = ic_n

        for idx, (resi, resj) in enumerate(
                zip(self.encode_res[:-1], self.encode_res[1:])):
            hidden_ch_i = min(ch_base // resi, ch_max)
            hidden_ch_j = min(ch_base // resj, ch_max)

            if idx == 0:
                block = EncoderBlock(hidden_ch_i, hidden_ch_j, rgb_n=ic_n, activation=activation)
            else:
                block = EncoderBlock(hidden_ch_i, hidden_ch_j, activation=activation)

            setattr(self, 'b{}'.format(resi), block)

        hidden_ch = min(ch_base // self.encode_res[-1], ch_max)
        self.b4 = EncoderBlock(hidden_ch, hidden_ch, activation=activation, down=1)

    def forward(self, img):
        x = None
        feats = {}
        for resi in self.encode_res[0:-1]:
            block = getattr(self, 'b{}'.format(resi))
            x, feat = block(x, img)
            feats[resi] = feat

        x, feat = self.b4(x, img)
        feats[4] = feat

        return x, feats


class SynthesisBlockFirst(nn.Module):
    def __init__(
        self,
        oc_n,
        resolution,
        rgb_n=None,
        activation=lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256),
    ):
        """
        Args:
            oc_n: output channel number
        """
        super().__init__()
        self.resolution = resolution

        self.conv1 = SeparableConv2d(oc_n, oc_n, 3, activation=activation)
        self.conv2 = SeparableConv2d(oc_n, oc_n, 3, resolution=4, activation=activation)

        if rgb_n is not None:
            self.torgb = nn.Conv2d(oc_n, rgb_n, 1)

    def forward(self, x, enc_feat):
        x = self.conv1(x)
        x = x + enc_feat
        x = self.conv2(x)

        img = None
        if self.torgb is not None:
            img = self.torgb(x)

        return x, img


class SynthesisBlock(nn.Module):
    def __init__(
        self,
        ic_n,
        oc_n,
        resolution,
        rgb_n,
        activation=lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256)
    ):
        super().__init__()

        self.resolution = resolution

        self.conv1 = SeparableConv2d(ic_n, oc_n, 3, resolution=resolution, up=2, activation=activation, use_noise=True)
        self.conv2 = SeparableConv2d(oc_n, oc_n, 3, resolution=resolution, up=1, activation=activation, use_noise=True)

        self.torgb = None
        if rgb_n is not None:
            self.torgb = nn.Conv2d(oc_n, rgb_n, 1)
        self.upsample = Upsample2d(rgb_n, resolution=resolution)

    def forward(self, x, enc_feat, img):
        x = self.conv1(x)
        x = x + enc_feat
        x = self.conv2(x)

        if img is not None:
            img = self.upsample(img)

        if self.torgb is not None:
            y = self.torgb(x)
            img = img.add_(y) if img is not None else y

        return x, img


class Synthesis(nn.Module):
    def __init__(
        self,
        resolution=256,
        rgb_n=3,
        ch_base=32768,
        ch_max=512,
        activation=lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256)
    ):
        super().__init__()

        log2res = int(np.log2(resolution))
        if 2 ** log2res != resolution:
            raise ValueError
        block_res = [2 ** i for i in range(2, log2res + 1)]

        self.resolution = resolution
        self.rgb_n = rgb_n
        self.block_res = block_res

        hidden_ch = min(ch_base // block_res[0], ch_max)
        self.b4 = SynthesisBlockFirst(hidden_ch, resolution=4, rgb_n=rgb_n, activation=activation)

        for resi, resj in zip(block_res[:-1], block_res[1:]):
            hidden_ch_i = min(ch_base // resi, ch_max)
            hidden_ch_j = min(ch_base // resj, ch_max)
            block = SynthesisBlock(hidden_ch_i, hidden_ch_j, resolution=resj, rgb_n=rgb_n, activation=activation)
            setattr(self, 'b{}'.format(resj), block)

    def forward(self, x, enc_feats):
        x, img = self.b4(x, enc_feats[4])
        for res in self.block_res[1:]:
            block = getattr(self, f'b{res}')
            x, img = block(x, enc_feats[res], img)
        return img


class Generator(nn.Module):
    def __init__(self, resolution=256):
        super().__init__()

        self.synthesis = Synthesis(resolution=resolution)
        self.encoder = Encoder(resolution=resolution)

    def forward(self, x):
        """
        Args:
            x: 4 channel rgb+mask
        """
        x, feats = self.encoder(x)
        img = self.synthesis(x, feats)
        return img
