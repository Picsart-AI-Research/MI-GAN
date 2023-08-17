from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

# Debug
# torch.use_deterministic_algorithms(True)

import numpy as np
import numpy.random as npr

from lib.model_zoo.common.get_model import get_model, register
from lib.model_zoo.common import utils

version = '2'
symbol = 'stylegan'

from torch_utils import misc
from torch_utils.ops import upfirdn2d, conv2d_gradfix, conv2d_resample, fma

from torch.nn.modules.utils import _pair


class conv2d(nn.Conv2d):
    """
    The slidely modified conv2d
    """

    def __init__(self, *args, **kwargs):
        use_wscale = kwargs.pop('use_wscale', False)
        super().__init__(*args, **kwargs)
        in_channels = args[0] if len(args) > 0 else kwargs['in_channels']
        kernel_size = args[2] if len(args) > 2 else kwargs['kernel_size']

        fan_in = in_channels * kernel_size * kernel_size

        he_std = 1 / np.sqrt(fan_in)
        if use_wscale:
            weight_init_std = 1
            self.weight_gain = he_std
        else:
            weight_init_std = he_std
            self.weight_gain = 1
        self.bias_gain = 1

        nn.init.normal_(self.weight, mean=0.0, std=weight_init_std)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        w_gain = self.weight_gain
        w = (self.weight * w_gain).to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        if self.padding_mode != 'zeros':
            return conv2d_gradfix.conv2d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                w, b, self.stride,
                _pair(0), self.dilation, self.groups)
        return conv2d_gradfix.conv2d(
            x, w, b, self.stride,
            self.padding, self.dilation, self.groups)


class dense(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 bias_init=0,
                 activation=None,
                 lr_multi=1, ):

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


def modulated_conv2d(x,
                     weight,
                     styles,
                     noise=None,
                     up=1,
                     down=1,
                     padding=0,
                     resample_filter=None,
                     demodulate=True,
                     flip_weight=True,
                     fused_modconv=True, ):
    """
    Args:
        x               : Input tensor of shape [batch_size, in_channels, in_height, in_width].
        weight          : Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
        styles          : Modulation coefficients of shape [batch_size, in_channels].
        noise           : Optional noise tensor to add to the output activations.
        up              : Integer upsampling factor.
        down            : Integer downsampling factor.
        padding         : Padding with respect to the upsampled image.
        resample_filter : Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
        demodulate      : Apply weight demodulation?
        flip_weight     : False = convolution, True = correlation (matches torch.nn.functional.conv2d).
        fused_modconv   : Perform modulation, convolution, and demodulation as a single fused operation?
    """

    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1, 2, 3],
                                                                            keepdim=True))  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None

    # Type StyleGan3 (Sg3)
    if demodulate:
        weight = weight * weight.square().mean([1, 2, 3], keepdim=True).rsqrt()
        styles = styles * styles.square().mean().rsqrt()

    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]

    if demodulate:
        # Original or Type StyleGan3 if activated
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]

        # Suspect that the collapse error is the overflow of square_sum
        # So make the following change (do norm first)

        # Type A
        # dcoefs = (w.square().mean(dim=[2,3,4])).rsqrt() / np.sqrt(np.prod(w.shape[2:]))

        # Type B
        # w_norminf = (w.norm(float('inf'), dim=[2,3,4]) + 1e-8).detach()
        # w_norminf_unsuqeezed = w_norminf.view(*w_norminf.shape, 1, 1, 1)
        # dcoefs = ((w/w_norminf_unsuqeezed).square().sum(dim=[2,3,4])).rsqrt()/w_norminf

    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down,
                                            padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding,
                                        groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


class conv2d_layer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 activation=None,
                 up=1,
                 down=1,
                 resample_filter=[1, 3, 3, 1], ):
        super().__init__()
        self.up = up
        self.down = down
        if resample_filter is not None:
            self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        else:
            self.resample_filter = None

        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.activation = None
        if activation is not None:
            self.activation = utils.get_unit()(activation)()

        w = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=torch.contiguous_format)
        self.weight = torch.nn.Parameter(w)
        self.bias = torch.nn.Parameter(torch.zeros([out_channels])) if bias else None

        self.repr = 'conv2d_layer({}, {}, kernal_size={}, bias={}, up={}, down={}, filter={}, act={})'.format(
            in_channels, out_channels, kernel_size, bias, up, down, filter, activation)

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        flip_weight = (self.up == 1)  # slightly faster
        x = conv2d_resample.conv2d_resample(
            x=x, w=w.to(x.dtype), f=self.resample_filter,
            up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1).to(x.dtype)
        if self.activation is not None:
            x = self.activation(x, gain=gain)
        else:
            x = x * gain
        return x

    def __repr__(self):
        return self.repr


class synthesis_layer(conv2d_layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 w_dim,
                 resolution,
                 bias=True,
                 activation='lrelu_agc(alpha=0.2, gain=sqrt_2)',
                 up=1,
                 resample_filter=[1, 3, 3, 1],
                 use_noise=True, ):

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            activation=activation,
            up=up,
            down=1,
            resample_filter=resample_filter)

        self.affine = dense(w_dim, in_channels, bias=True, bias_init=1, activation=None)
        self.resolution = resolution
        self.use_noise = use_noise
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.repr = 'synthesis_layer({}, {}, kernal_size={}, bias={}, up={}, down={}, filter={}, act={}, noise={})'.format(
            in_channels, out_channels, kernel_size, bias, up, 1, filter, activation, use_noise)

    def forward(self, x, w, fused_modconv=True, gain=1, noise_mode='random'):

        assert noise_mode in ['random', 'const', 'none']
        styles = self.affine(w)  # The original StyleGan-ADA code
        # styles = self.affine(w)+1 # The add one that only happened in CoModGan original code
        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution],
                                device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1)  # slightly faster

        # weight = self.weight * self.weight_gain
        # This mimic the original TensorFlow Code
        # However, in modulated conv2d, the demodulation is active,
        #     so it maybe unnecessary to do any reweight in weight
        #     because it will be wipe out anyway.
        # Reference: https://paperswithcode.com/method/weight-demodulation
        x = modulated_conv2d(
            x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight,
            fused_modconv=fused_modconv)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1).to(x.dtype)
        if self.activation is not None:
            x = self.activation(x, gain=gain)
        else:
            x = x * gain
        return x


class torgb_layer(conv2d_layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 w_dim,
                 activation=None, ):

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            bias=True,
            activation=activation,
            up=1,
            down=1,
            resample_filter=None)
        self.affine = dense(w_dim, in_channels, bias=True, bias_init=1, activation=None)

    def forward(self, x, w, fused_modconv=True):
        # styles = self.affine(w)
        # weight = self.weight * self.weight_gain
        # x = modulated_conv2d(x=x, weight=weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        # This is to mimic the original TF code ^^^
        # Originally it looks like this
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1).to(x.dtype)
        if self.activation is not None:
            x = self.activation(x)
        return x


#############
# G_mapping #
#############

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


@register('stylegan2_mapping', version)
class Mapping(nn.Module):
    def __init__(self,
                 z_dim=512,
                 c_dim=0,
                 w_dim=512,
                 num_ws=14,
                 num_layers=8,
                 embed_features=None,
                 layer_features=None,
                 activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
                 lr_multiplier=0.01,
                 w_avg_beta=0.995, ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim

        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = dense(
                c_dim, embed_features, act=None)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = dense(
                in_features, out_features,
                activation=activation,
                lr_multi=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

        # XX_Debug
        # self.load_state_dict(torch.load('data/debug/mapping.pth'))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):

        # XX_Debug
        # fixseed(0)
        # z = torch.randn_like(z)

        # Embed, normalize, and concat inputs.
        x = None
        if self.z_dim > 0:
            x = normalize_2nd_moment(z.to(torch.float32))
        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)
            # XX_debug
            # dp(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            assert self.w_avg_beta is not None
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x


#############
# Generator #
#############

class synthesis_block(nn.Module):
    def __init__(self,
                 ic_n,
                 oc_n,
                 w_dim,
                 resolution,
                 rgb_n=None,
                 resample_filter=[1, 3, 3, 1],
                 activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
                 res_link=False,
                 use_fp16=False, ):

        super().__init__()
        self.w_dim = w_dim
        self.resolution = resolution
        self.use_fp16 = use_fp16
        self.res_link = res_link
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_conv = 0
        self.num_torgb = 0

        self.const = None
        self.conv0 = None
        if ic_n == 0:
            self.const = torch.nn.Parameter(torch.randn([oc_n, resolution, resolution]))
        else:
            self.conv0 = synthesis_layer(
                ic_n, oc_n, 3, w_dim=w_dim,
                resolution=resolution, up=2, activation=activation,
                resample_filter=resample_filter, use_noise=True)
            self.num_conv += 1

        self.conv1 = synthesis_layer(
            oc_n, oc_n, 3, w_dim=w_dim,
            resolution=resolution, up=1, activation=activation,
            resample_filter=None, use_noise=True)
        self.num_conv += 1

        self.torgb = None
        if rgb_n is not None:
            self.torgb = torgb_layer(oc_n, rgb_n, 1, w_dim=w_dim, activation=None)
            self.num_torgb += 1

        if ic_n != 0 and res_link:
            self.skip = conv2d_layer(
                ic_n, oc_n, kernel_size=1, bias=False, up=2, down=1,
                resample_filter=resample_filter, )

    def forward(self, x, img, ws, fused_modconv=None, noise_mode='random'):
        dtype = torch.float16 if self.use_fp16 else torch.float32

        if fused_modconv is None:
            with misc.suppress_tracer_warnings():
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        if self.const is not None:
            x = self.const.to(dtype=dtype, memory_format=torch.contiguous_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            x = x.to(dtype=dtype, memory_format=torch.contiguous_format)

        if self.res_link:
            y = self.skip(x, gain=np.sqrt(0.5))

        w_iter = iter(ws.unbind(dim=1))

        if self.conv0 is not None:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, noise_mode=noise_mode)

        if self.res_link:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), noise_mode=noise_mode)
            x = y.add_(x)
        else:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, noise_mode=noise_mode)

        if img is not None:
            img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.torgb is not None:
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        return x, img


@register('stylegan2_synthesis', version)
class Synthesis(nn.Module):
    def __init__(self,
                 w_dim=512,
                 resolution=256,
                 rgb_n=3,
                 ch_base=16384,
                 ch_max=512,
                 use_fp16_after_res=16,
                 resample_filter=[1, 3, 3, 1],
                 activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)', ):

        super().__init__()

        log2res = int(np.log2(resolution))
        if 2 ** log2res != resolution:
            raise ValueError
        block_res = [2 ** i for i in range(2, log2res + 1)]

        self.w_dim = w_dim
        self.resolution = resolution
        self.rgb_n = rgb_n
        self.block_res = block_res

        self.num_ws = 0
        for resi, resj in zip([None] + block_res[:-1], block_res):
            hidden_ch_i = min(ch_base // resi, ch_max) if resi is not None else 0
            hidden_ch_j = min(ch_base // resj, ch_max)
            use_fp16 = (resj > use_fp16_after_res)
            block = synthesis_block(
                hidden_ch_i, hidden_ch_j,
                w_dim=w_dim,
                resolution=resj,
                rgb_n=rgb_n,
                resample_filter=resample_filter,
                activation=activation,
                res_link=False,
                use_fp16=use_fp16, )
            self.num_ws += block.num_conv
            if resj == block_res[-1]:
                self.num_ws += block.num_torgb
            setattr(self, 'b{}'.format(resj), block)

    def forward(self, ws, noise_mode='random'):
        block_ws = []
        ws = ws.to(torch.float32)
        w_idx = 0
        for res in self.block_res:
            block = getattr(self, f'b{res}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_res, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, noise_mode=noise_mode)
        return img


@register('stylegan2_generator', version)
class Generator(nn.Module):
    def __init__(self,
                 mapping,
                 synthesis, ):
        super().__init__()
        if isinstance(mapping, nn.Module):
            self.mapping = mapping
        else:
            self.mapping = get_model()(mapping)
        if isinstance(synthesis, nn.Module):
            self.synthesis = synthesis
        else:
            self.synthesis = get_model()(synthesis)
        if self.synthesis.num_ws != self.mapping.num_ws:
            raise ValueError
        self.num_ws = self.mapping.num_ws
        self.z_dim = self.mapping.z_dim
        self.c_dim = self.mapping.c_dim
        self.w_dim = self.mapping.w_dim
        self.img_resolution = self.synthesis.resolution
        self.img_channels = self.synthesis.rgb_n

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img


#################
# Discriminator #
#################

def compute_r1_penalty_from_outputs(d_outputs, x_real):
    """
    Computes R1 grad penalty based on the existing d_outputs to save memory
    This is get from INR GAN
    """
    r1_grads = torch.autograd.grad(
        outputs=[d_outputs.sum()], inputs=[x_real],
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    r1_penalty = r1_grads.square().sum([1, 2, 3])
    return r1_penalty


class discrim_block(nn.Module):
    def __init__(self,
                 ic_n,
                 mc_n,
                 oc_n,
                 rgb_n=None,
                 resample_filter=[1, 3, 3, 1],
                 activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
                 reslink=False,
                 use_fp16=False, ):

        super().__init__()
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.fromrgb = None
        if rgb_n is not None:
            self.fromrgb = conv2d_layer(
                rgb_n, mc_n, 1, bias=True,
                activation=activation, up=1, down=1, resample_filter=None)

        self.conv0 = conv2d_layer(
            ic_n, mc_n, 3, bias=True,
            activation=activation, up=1, down=1, resample_filter=None)
        self.conv1 = conv2d_layer(
            mc_n, oc_n, 3, bias=True,
            activation=activation, up=1, down=2, resample_filter=resample_filter)

        self.reslink = reslink
        if reslink:
            self.skip = conv2d_layer(
                mc_n, oc_n, 1, bias=False,
                activation=None, up=1, down=2, resample_filter=resample_filter)
        self.use_fp16 = use_fp16

    def forward(self, x, img):
        if x is not None:
            if self.use_fp16:
                x = x.to(dtype=torch.float16)
            else:
                x = x.to(dtype=torch.float32)

        if self.fromrgb is not None:
            if self.use_fp16:
                img = img.to(dtype=torch.float16)
            else:
                img = img.to(dtype=torch.float32)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            # img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None
        img = None

        if self.reslink:
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

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


class discrim_epilogue(nn.Module):
    def __init__(self,
                 ic_n,
                 resolution,
                 cmap_dim,
                 rgb_n=None,
                 mbstd_group_size=4,
                 mbstd_c_n=1,
                 activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
                 reslink=True, ):

        super().__init__()
        self.ic_n = ic_n
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.rgb_n = rgb_n
        self.reslink = reslink

        self.fromrgb = None
        if rgb_n is not None:
            self.fromrgb = conv2d_layer(
                rgb_n, ic_n, 1, bias=True,
                activation=activation, up=1, down=1, resample_filter=None)

        self.mbstd = None
        if mbstd_c_n > 0:
            self.mbstd = minibatch_std_layer(
                group_size=mbstd_group_size,
                num_channels=mbstd_c_n)

        self.conv = conv2d_layer(
            ic_n + mbstd_c_n, ic_n, 3, bias=True,
            activation=activation, up=1, down=1, resample_filter=None)
        self.fc = dense(ic_n * (resolution ** 2), ic_n, activation=activation)
        self.out = dense(ic_n, 1 if cmap_dim is None else cmap_dim, activation=None)

    def forward(self, x, img=None, cmap=None):
        x = x.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        if self.fromrgb is not None:
            img = img.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            x = x + self.fromrgb(img)
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)
        if self.cmap_dim is not None:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        return x


@register('stylegan2_discriminator', version)
class Discriminator(nn.Module):
    def __init__(self,
                 resolution=256,
                 ic_n=3,
                 ch_base=16384,
                 ch_max=512,
                 use_fp16_before_res=16,
                 resample_filter=[1, 3, 3, 1],
                 activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
                 mbstd_group_size=4,
                 mbstd_c_n=1,
                 c_dim=None,
                 cmap_dim=None, ):

        super().__init__()

        log2res = int(np.log2(resolution))
        if 2 ** log2res != resolution:
            raise ValueError
        self.encode_res = [2 ** i for i in range(log2res, 1, -1)]
        self.ic_n = ic_n
        self.ch_base = ch_base
        self.ch_max = ch_max
        self.resample_filter = resample_filter
        self.activation = activation

        for idx, (resi, resj) in enumerate(
                zip(self.encode_res[:-1], self.encode_res[1:])):
            hidden_ch_i = min(ch_base // resi, ch_max)
            hidden_ch_j = min(ch_base // resj, ch_max)
            use_fp16 = False if use_fp16_before_res is None else (resi > use_fp16_before_res)

            if idx == 0:
                block = discrim_block(
                    hidden_ch_i, hidden_ch_i, hidden_ch_j,
                    rgb_n=ic_n,
                    resample_filter=resample_filter,
                    activation=activation,
                    reslink=True,
                    use_fp16=use_fp16, )
            else:
                block = discrim_block(
                    hidden_ch_i, hidden_ch_i, hidden_ch_j,
                    rgb_n=None,
                    resample_filter=resample_filter,
                    activation=activation,
                    reslink=True,
                    use_fp16=use_fp16, )

            setattr(self, 'b{}'.format(resi), block)

        self.mapping = None
        if (c_dim is not None) and (c_dim > 0):
            self.mapping = Mapping(
                z_dim=0,
                c_dim=c_dim,
                w_dim=cmap_dim,
                num_ws=None,
                w_avg_beta=None, )

        hidden_ch = min(ch_base // self.encode_res[-1], ch_max)
        self.b4 = discrim_epilogue(
            hidden_ch,
            resolution=4,
            cmap_dim=None,
            activation=activation,
            mbstd_group_size=mbstd_group_size,
            mbstd_c_n=mbstd_c_n,
        )

    def forward(self, img, c, **kwargs):
        x = None
        for resi in self.encode_res[0:-1]:
            block = getattr(self, 'b{}'.format(resi))
            x, img = block(x, img)

        cmap = None
        if self.mapping is not None:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x


#########
# debug #
#########

def dp(v):
    vsum = np.abs(v.cpu().detach().numpy().astype(float)).sum()
    print(vsum, v.dtype, v.shape)
    return vsum


def fixseed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def profile(module, input, nb_iters=1000):
    input = input.detach()
    import time
    # Warmup
    for _ in range(nb_iters // 10):
        output = module(input)
        output.sum().backward()
        for param in module.parameters():
            param.grad = None

    for _ in range(nb_iters):
        ckpt = time.time()
        output = module(input)
        ckpt_fwd = time.time() - ckpt;
        ckpt = time.time()
        output.sum().backward()
        ckpt_bwd = time.time() - ckpt;
        ckpt = time.time()
        for param in module.parameters():
            param.grad = None
    print(ckpt_fwd / nb_iters, ckpt_bwd / nb_iters)
