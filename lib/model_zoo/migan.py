import numpy as np
import torch
import torch.nn as nn

from torch_utils.ops import upfirdn2d, conv2d_resample

from lib.model_zoo.common.get_model import get_model, register
from lib.model_zoo.common import utils

version = '1'
symbol = 'migan_inpainting'


class Dense(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        bias_init=0,
        activation=None,
        lr_multi=1
    ):
        super().__init__()

        self.activation = None
        if activation is not None:
            self.activation = utils.get_unit()(activation)()

        self.weight = nn.Parameter(torch.randn([out_features, in_features]) / lr_multi)
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multi / np.sqrt(in_features)
        self.bias_gain = lr_multi
        self.repr = 'dense({}, {}, bias={}, act={}, lr_multi={})'.format(
            in_features, out_features, bias, activation, lr_multi)

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None:
            if self.bias_gain != 1:
                b = b * self.bias_gain
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = torch.mm(x, w.t())
        if self.activation is not None:
            x = self.activation(x)
        return x

    def __repr__(self):
        return self.repr


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        activation=None,
        up=1,
        down=1,
        resample_filter=None,
        resolution=None,
        use_noise=False,
        reparametrize=False,
        num_reparam_tensors=4,
        groups=1
    ):
        super().__init__()
        self.up = up
        self.down = down
        self.reparametrize = reparametrize
        self.num_reparam_tensors = num_reparam_tensors
        if resample_filter is not None:
            self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        else:
            self.resample_filter = None

        self.padding = kernel_size // 2
        self.activation = None
        if activation is not None:
            self.activation = utils.get_unit()(activation)()

        if reparametrize:
            for i in range(num_reparam_tensors):
                w = torch.randn([out_channels, in_channels // groups, kernel_size, kernel_size]).to(memory_format=torch.contiguous_format)
                setattr(self, f"w{i}", torch.nn.Parameter(w))
        else:
            w = torch.randn([out_channels, in_channels // groups, kernel_size, kernel_size]).to(memory_format=torch.contiguous_format)
            self.weight = torch.nn.Parameter(w)
        self.bias = torch.nn.Parameter(torch.zeros([out_channels])) if bias else None

        self.use_noise = use_noise
        if use_noise:
            assert resolution is not None
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.repr = 'Conv2d({}, {}, kernal_size={}, bias={}, up={}, down={}, act={}, noise={}, groups={}'.format(
            in_channels, out_channels, kernel_size, bias, up, down, activation, use_noise, groups)
        self.groups = groups

    def forward(self, x, gain=1, noise_mode='none'):
        assert noise_mode in ['random', 'const', 'none']

        if self.reparametrize:
            w = getattr(self, "w0")
            for i in range(1, self.num_reparam_tensors):
                w = w + getattr(self, f"w{i}")
            w = w / np.sqrt(self.num_reparam_tensors)
        else:
            w = self.weight
        w = w * (w.square().sum(dim=[1, 2, 3]) + 1e-8).rsqrt().reshape(-1, 1, 1, 1)  # Normalize weights

        flip_weight = (self.up == 1)  # slightly faster

        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            flip_weight=flip_weight,
            groups=self.groups
        )

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]],
                                device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        if noise is not None:
            x = x.add_(noise)

        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1).to(x.dtype)
        if self.activation is not None:
            x = self.activation(x, gain=gain)
        else:
            x = x * gain
        return x

    def __repr__(self):
        return self.repr


class SeparableConv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        bias=True,
        activation=None,
        up=1,
        down=1,
        resample_filter=None,
        resolution=None,
        use_noise=False,
        reparametrize=False,
        num_reparam_tensors=4,
    ):
        super().__init__()

        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            bias=bias,
            activation=activation,
            up=1,
            down=1,
            resample_filter=None,
            resolution=resolution,
            use_noise=False,
            groups=in_channels,
            reparametrize=reparametrize,
            num_reparam_tensors=num_reparam_tensors,
        )
        self.conv2 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            activation=activation,
            up=up,
            down=down,
            resample_filter=resample_filter,
            resolution=resolution,
            use_noise=use_noise,
            groups=1,
            reparametrize=reparametrize,
            num_reparam_tensors=num_reparam_tensors,
        )

    def forward(self, x, gain=1, noise_mode='none'):
        x = self.conv1(x, gain=gain, noise_mode='none')
        x = self.conv2(x, gain=gain, noise_mode=noise_mode)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        ic_n,
        oc_n,
        rgb_n=None,
        resample_filter=[1, 3, 3, 1],
        activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
        down=2,
        depthwise=False,
        reparametrize=False,
        num_reparam_tensors=4,
    ):
        super().__init__()

        self.fromrgb = None
        if rgb_n is not None:
            self.fromrgb = Conv2d(rgb_n, ic_n, 1, activation=activation)

        if depthwise:
            self.conv1 = SeparableConv2d(
                ic_n, ic_n, 3, activation=activation,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )
            self.conv2 = SeparableConv2d(
                ic_n, oc_n, 3, activation=activation, down=down,
                resample_filter=resample_filter if down is not None else None,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )
        else:
            self.conv1 = Conv2d(ic_n, ic_n, 3, activation=activation,
                                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors)
            self.conv2 = Conv2d(
                ic_n, oc_n, 3, activation=activation, down=down,
                resample_filter=resample_filter if down is not None else None,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )

    def forward(self, x, img):
        if x is not None:
            x = x.to(dtype=torch.float32)

        if self.fromrgb is not None:
            img = img.to(dtype=torch.float32)
            y = self.fromrgb(img)
            x = x + y if x is not None else y

        feat = self.conv1(x)
        x = self.conv2(feat)
        return x, feat


@register('migan_encoder', version)
class Encoder(nn.Module):
    def __init__(
        self,
        resolution=256,
        ic_n=3,
        ch_base=32768,
        ch_max=512,
        resample_filter=[1, 3, 3, 1],
        activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
        depthwise=False,
        reparametrize=False,
        num_reparam_tensors=4
    ):
        super().__init__()

        log2res = int(np.log2(resolution))
        if 2 ** log2res != resolution:
            raise ValueError
        self.encode_res = [2**i for i in range(log2res, 1, -1)]
        self.ic_n = ic_n

        for idx, (resi, resj) in enumerate(
                zip(self.encode_res[:-1], self.encode_res[1:])):
            hidden_ch_i = min(ch_base // resi, ch_max)
            hidden_ch_j = min(ch_base // resj, ch_max)

            if idx == 0:
                block = EncoderBlock(
                    hidden_ch_i, hidden_ch_j,
                    rgb_n=ic_n,
                    resample_filter=resample_filter,
                    activation=activation,
                    depthwise=depthwise,
                    reparametrize=reparametrize,
                    num_reparam_tensors=num_reparam_tensors
                )
            else:
                block = EncoderBlock(
                    hidden_ch_i, hidden_ch_j,
                    resample_filter=resample_filter,
                    activation=activation,
                    depthwise=depthwise,
                    reparametrize=reparametrize,
                    num_reparam_tensors=num_reparam_tensors
                )

            setattr(self, 'b{}'.format(resi), block)

        hidden_ch = min(ch_base // self.encode_res[-1], ch_max)
        self.b4 = EncoderBlock(
            hidden_ch,
            hidden_ch,
            activation=activation,
            down=1,
            depthwise=depthwise,
            reparametrize=reparametrize,
            num_reparam_tensors=num_reparam_tensors
        )

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
        activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
        depthwise=False,
        reparametrize=False,
        num_reparam_tensors=4
    ):
        """
        Args:
            oc_n: output channel number
        """
        super().__init__()
        self.resolution = resolution
        if depthwise:
            self.conv1 = SeparableConv2d(
                oc_n, oc_n, 3, activation=activation,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )
            self.conv2 = SeparableConv2d(
                oc_n, oc_n, 3, resolution=4, activation=activation,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )
        else:
            self.conv1 = Conv2d(
                oc_n, oc_n, 3, activation=activation,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )
            self.conv2 = Conv2d(
                oc_n, oc_n, 3, resolution=4, activation=activation,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )

        if rgb_n is not None:
            self.torgb = Conv2d(
                oc_n, rgb_n, 1, activation=None,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )

    def forward(self, x, enc_feat, noise_mode='random'):
        dtype = torch.float32

        x = x .to(dtype=dtype, memory_format=torch.contiguous_format)
        enc_feat = enc_feat.to(dtype=dtype, memory_format=torch.contiguous_format)

        x = self.conv1(x)
        x = x + enc_feat
        x = self.conv2(x, noise_mode=noise_mode)

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
        resample_filter=[1, 3, 3, 1],
        activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
        depthwise=False,
        reparametrize=False,
        num_reparam_tensors=4
    ):
        super().__init__()

        self.resolution = resolution
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        if depthwise:
            self.conv1 = SeparableConv2d(
                ic_n, oc_n, 3,
                resolution=resolution, up=2, activation=activation,
                resample_filter=resample_filter, use_noise=True,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )

            self.conv2 = SeparableConv2d(
                oc_n, oc_n, 3,
                resolution=resolution, up=1, activation=activation, use_noise=True,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )
        else:
            self.conv1 = Conv2d(
                ic_n, oc_n, 3,
                resolution=resolution, up=2, activation=activation,
                resample_filter=resample_filter, use_noise=True,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )

            self.conv2 = Conv2d(
                oc_n, oc_n, 3,
                resolution=resolution, up=1, activation=activation, use_noise=True,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )

        self.torgb = None
        if rgb_n is not None:
            self.torgb = Conv2d(
                oc_n, rgb_n, 1, activation=None,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )

    def forward(self, x, enc_feat, img, noise_mode='random'):
        dtype = torch.float32

        x = x .to(dtype=dtype, memory_format=torch.contiguous_format)
        enc_feat = enc_feat.to(dtype=dtype, memory_format=torch.contiguous_format)

        x = self.conv1(x, noise_mode=noise_mode)
        x = x + enc_feat
        x = self.conv2(x, noise_mode=noise_mode)

        if img is not None:
            img = upfirdn2d.upsample2d(img, self.resample_filter)

        to_rgb_out = None
        if self.torgb is not None:
            y = self.torgb(x)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            to_rgb_out = y
            img = img.add_(y) if img is not None else y

        return x, img, to_rgb_out


@register('migan_synthesis', version)
class Synthesis(nn.Module):
    def __init__(
        self,
        resolution=256,
        rgb_n=3,
        ch_base=32768,
        ch_max=512,
        resample_filter=[1, 3, 3, 1],
        activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
        depthwise=False,
        reparametrize=False,
        num_reparam_tensors=4
    ):
        super().__init__()

        log2res = int(np.log2(resolution))
        if 2 ** log2res != resolution:
            raise ValueError
        block_res = [2 ** i for i in range(2, log2res+1)]

        self.resolution = resolution
        self.rgb_n = rgb_n
        self.block_res = block_res

        hidden_ch = min(ch_base // block_res[0], ch_max)
        self.b4 = SynthesisBlockFirst(
            hidden_ch, resolution=4,
            rgb_n=rgb_n, activation=activation, depthwise=depthwise,
            reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
        )

        for resi, resj in zip(block_res[:-1], block_res[1:]):
            hidden_ch_i = min(ch_base // resi, ch_max)
            hidden_ch_j = min(ch_base // resj, ch_max)
            block = SynthesisBlock(
                hidden_ch_i, hidden_ch_j,
                resolution=resj,
                rgb_n=rgb_n,
                resample_filter=resample_filter,
                activation=activation,
                depthwise=depthwise,
                reparametrize=reparametrize,
                num_reparam_tensors=num_reparam_tensors
            )

            setattr(self, 'b{}'.format(resj), block)

    def forward(self, x, enc_feats, noise_mode='random'):
        x, img = self.b4(x, enc_feats[4], noise_mode=noise_mode)
        intermediate_outputs = {"res_to_rgb": {4: img}, "res_img": {4: img}}
        for res in self.block_res[1:]:
            block = getattr(self, f'b{res}')
            x, img, to_rgb_out = block(x, enc_feats[res], img, noise_mode=noise_mode)
            intermediate_outputs["res_to_rgb"][res] = to_rgb_out
            intermediate_outputs["res_img"][res] = img
        return img, intermediate_outputs


@register('migan_generator', version)
class Generator(nn.Module):
    def __init__(self, encoder, synthesis):
        super().__init__()

        if isinstance(synthesis, nn.Module):
            self.synthesis = synthesis
        else:
            self.synthesis = get_model()(synthesis)

        self.img_resolution = self.synthesis.resolution
        self.img_channels = self.synthesis.rgb_n

        if isinstance(encoder, nn.Module):
            self.encoder = encoder
        else:
            self.encoder = get_model()(encoder)
        self.ic_n = self.encoder.ic_n

    def forward(self, x, noise_mode='random', return_intermediate_outputs=False):
        """
        Args:
            x: 4 channel rgb+mask
        """
        x, feats = self.encoder(x)
        img, intermediate_outputs = self.synthesis(x, feats, noise_mode=noise_mode)
        if return_intermediate_outputs:
            return img, intermediate_outputs
        return img

#################
# Discriminator #
#################


class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        ic_n,
        oc_n,
        rgb_n=None,
        resample_filter=[1, 3, 3, 1],
        activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
        depthwise=False,
        reparametrize=False,
        num_reparam_tensors=4
    ):
        super().__init__()

        self.fromrgb = None
        if rgb_n is not None:
            self.fromrgb = Conv2d(
                rgb_n, ic_n, 1, activation=activation,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )

        if depthwise:
            self.conv1 = SeparableConv2d(
                ic_n, ic_n, 3, activation=activation,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )
            self.conv2 = SeparableConv2d(
                ic_n, oc_n, 3, activation=activation, down=2, resample_filter=resample_filter,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )
        else:
            self.conv1 = Conv2d(
                ic_n, ic_n, 3, activation=activation,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )
            self.conv2 = Conv2d(
                ic_n, oc_n, 3, activation=activation, down=2, resample_filter=resample_filter,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )
        self.skip = Conv2d(
            ic_n, oc_n, 1, bias=False, activation=None, down=2, resample_filter=resample_filter,
            reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
        )

    def forward(self, x, img):
        if x is not None:
            x = x.to(dtype=torch.float32)

        if self.fromrgb is not None:
            img = img.to(dtype=torch.float32)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
        img = None

        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv1(x)
        x = self.conv2(x, gain=np.sqrt(0.5))
        x = y.add_(x)

        return x, img


class minibatch_std_layer(nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F
        y = x.reshape(G, -1, F, c, H,
                      W)  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


class DiscriminatorEpilogue(nn.Module):
    def __init__(
        self,
        ic_n,
        resolution,
        mbstd_group_size=4,
        mbstd_c_n=1,
        activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
        depthwise=False,
        reparametrize=False,
        num_reparam_tensors=4
    ):
        super().__init__()

        self.ic_n = ic_n
        self.resolution = resolution

        self.mbstd = None
        if mbstd_c_n > 0:
            self.mbstd = minibatch_std_layer(
                group_size=mbstd_group_size,
                num_channels=mbstd_c_n)

        if depthwise:
            self.conv = SeparableConv2d(
                ic_n + mbstd_c_n, ic_n, 3, activation=activation,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )
        else:
            self.conv = Conv2d(
                ic_n + mbstd_c_n, ic_n, 3, activation=activation,
                reparametrize=reparametrize, num_reparam_tensors=num_reparam_tensors
            )

        self.fc = Dense(ic_n * (resolution ** 2), ic_n, activation=activation)
        self.out = Dense(ic_n, 1, activation=None)

    def forward(self, x):
        x = x.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)
        return x


@register('migan_discriminator', version)
class Discriminator(nn.Module):
    def __init__(
        self,
        resolution=256,
        ic_n=3,
        ch_base=32768,
        ch_max=512,
        resample_filter=[1, 3, 3, 1],
        activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
        mbstd_group_size=4,
        mbstd_c_n=1,
        depthwise=False,
        reparametrize=False,
        num_reparam_tensors=4
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
                block = DiscriminatorBlock(
                    hidden_ch_i, hidden_ch_j,
                    rgb_n=ic_n,
                    resample_filter=resample_filter,
                    activation=activation,
                    depthwise=depthwise,
                    reparametrize=reparametrize,
                    num_reparam_tensors=num_reparam_tensors
                )
            else:
                block = DiscriminatorBlock(
                    hidden_ch_i, hidden_ch_j,
                    rgb_n=None,
                    resample_filter=resample_filter,
                    activation=activation,
                    depthwise=depthwise,
                    reparametrize=reparametrize,
                    num_reparam_tensors=num_reparam_tensors
                )

            setattr(self, 'b{}'.format(resi), block)

        hidden_ch = min(ch_base // self.encode_res[-1], ch_max)
        self.b4 = DiscriminatorEpilogue(
            hidden_ch,
            resolution=4,
            activation=activation,
            mbstd_group_size=mbstd_group_size,
            mbstd_c_n=mbstd_c_n,
            depthwise=depthwise,
            reparametrize=reparametrize,
            num_reparam_tensors=num_reparam_tensors
        )

    def forward(self, img):
        x = None
        for resi in self.encode_res[0:-1]:
            block = getattr(self, 'b{}'.format(resi))
            x, img = block(x, img)
        x = self.b4(x)
        return x
