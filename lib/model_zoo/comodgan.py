import torch
import torch.nn as nn

# Debug
# torch.use_deterministic_algorithms(True)

import numpy as np

from lib.model_zoo.common.get_model import get_model, register

version = '3'
symbol = 'comodgan'

from torch_utils import misc
from torch_utils.ops import upfirdn2d

from .stylegan import Mapping as Mapping_StyleGan
from .stylegan import Discriminator as Discriminator_StyleGan
from .stylegan import Generator as Generator_StyleGan

from .stylegan import dense, synthesis_layer, torgb_layer
from .stylegan import synthesis_block as stylegan_synthesis_block
from .stylegan import discrim_block, discrim_epilogue


@register('comodgan_mapping')
class Mapping(Mapping_StyleGan):
    pass


class encoder_block(discrim_block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            feat = self.conv0(x)
            x = self.conv1(feat, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            feat = self.conv0(x)
            x = self.conv1(feat)
        return x, img, feat


class encoder_epilogue(discrim_epilogue):
    def __init__(self,
                 ic_n,
                 oc_n,
                 resolution,
                 cmap_dim,
                 rgb_n=None,
                 mbstd_group_size=4,
                 mbstd_c_n=1,
                 activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
                 reslink=True,
                 use_dropout=True,
                 has_extra_final_layer=True):
        super().__init__(
            ic_n=ic_n,
            resolution=resolution,
            cmap_dim=cmap_dim,
            rgb_n=rgb_n,
            mbstd_group_size=mbstd_group_size,
            mbstd_c_n=mbstd_c_n,
            activation=activation,
            reslink=reslink, )

        self.fc = dense(ic_n * (resolution ** 2), oc_n, activation=activation)
        if has_extra_final_layer:
            self.out = dense(oc_n, oc_n, activation=None)
        else:
            self.out = None
        self.dropout = None
        if use_dropout:
            self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, img=None, cmap=None):
        x = x.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        if self.fromrgb is not None:
            img = img.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            x = x + self.fromrgb(img)
        if self.mbstd is not None:
            x = self.mbstd(x)
        feat = self.conv(x)
        x = self.fc(feat.flatten(1))
        if self.out is not None:
            x = self.out(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.cmap_dim is not None:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        return x, feat


@register('comodgan_encoder', version)
class Encoder(nn.Module):
    def __init__(self,
                 resolution=256,
                 ic_n=4,
                 oc_n=1024,
                 ch_base=32768,
                 ch_max=512,
                 use_fp16_before_res=None,
                 resample_filter=[1, 3, 3, 1],
                 activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
                 mbstd_group_size=0,
                 mbstd_c_n=0,
                 c_dim=None,
                 cmap_dim=None,
                 use_dropout=True,
                 has_extra_final_layer=False):

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
                block = encoder_block(
                    hidden_ch_i, hidden_ch_i, hidden_ch_j,
                    rgb_n=ic_n,
                    resample_filter=resample_filter,
                    activation=activation,
                    reslink=False,
                    use_fp16=use_fp16, )
            else:
                block = encoder_block(
                    hidden_ch_i, hidden_ch_i, hidden_ch_j,
                    rgb_n=None,
                    resample_filter=resample_filter,
                    activation=activation,
                    reslink=False,
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
        self.b4 = encoder_epilogue(
            hidden_ch, oc_n,
            resolution=4,
            cmap_dim=None,
            activation=activation,
            mbstd_group_size=mbstd_group_size,
            mbstd_c_n=mbstd_c_n,
            reslink=False,
            use_dropout=use_dropout,
            has_extra_final_layer=has_extra_final_layer)

    def forward(self, img, c=None):
        x = None
        feats = {}
        for resi in self.encode_res[0:-1]:
            block = getattr(self, 'b{}'.format(resi))
            x, img, feat = block(x, img)
            feats[resi] = feat

        cmap = None
        if self.mapping is not None:
            cmap = self.mapping(None, c)
        x, feat = self.b4(x, img, cmap)
        feats[4] = feat

        return x, feats


class synthesis_block_first(nn.Module):
    def __init__(self,
                 w0_dim,
                 oc_n,
                 w_dim,
                 resolution,
                 rgb_n=None,
                 activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)', ):

        """
        Args:
            w0_dim: the bottleneck comodulate vector (x_global)
            oc_n: output channel number
            w_dim:  the middle code dimention.
        """

        super().__init__()
        self.resolution = resolution
        self.fc = dense(w0_dim, oc_n * (resolution ** 2), activation=activation)

        self.num_conv = 0
        self.num_torgb = 0

        self.conv = synthesis_layer(oc_n, oc_n, 3, w0_dim + w_dim, resolution=4, bias=True, activation=activation)
        self.num_conv += 1

        if rgb_n is not None:
            self.torgb = torgb_layer(oc_n, rgb_n, 1, w0_dim + w_dim, activation=None)
            self.num_torgb += 1

    def forward(self, x, x0, ws, fused_modconv=None, noise_mode='random'):
        dtype = torch.float32

        if fused_modconv is None:
            with misc.suppress_tracer_warnings():
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        x = x.to(dtype=dtype, memory_format=torch.contiguous_format)
        x0 = x0.to(dtype=dtype, memory_format=torch.contiguous_format)
        w0 = x

        x = self.fc(x)
        x = x.view(x.size(0), -1, self.resolution, self.resolution)
        x = x + x0

        w_iter = iter(ws.unbind(dim=1))

        w_long = torch.cat([next(w_iter), w0], dim=1)
        x = self.conv(x, w_long, fused_modconv=fused_modconv, noise_mode=noise_mode)

        img = None
        if self.torgb is not None:
            w_long = torch.cat([next(w_iter), w0], dim=1)
            img = self.torgb(x, w_long, fused_modconv=True)  # TODO: check whether True is correct???

        return x, img


class synthesis_block(stylegan_synthesis_block):
    def __init__(self,
                 ic_n,
                 oc_n,
                 w_dim,
                 w0_dim,
                 resolution,
                 rgb_n,
                 resample_filter=[1, 3, 3, 1],
                 activation='lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
                 res_link=False,
                 use_fp16=False, ):

        if ic_n == 0:
            raise ValueError

        super().__init__(
            ic_n,
            oc_n,
            w_dim,
            resolution,
            rgb_n,
            resample_filter,
            activation,
            res_link,
            use_fp16, )

        self.conv0 = synthesis_layer(
            ic_n, oc_n, 3, w_dim=w_dim + w0_dim,
            resolution=resolution, up=2, activation=activation,
            resample_filter=resample_filter, use_noise=True)

        self.conv1 = synthesis_layer(
            oc_n, oc_n, 3, w_dim=w_dim + w0_dim,
            resolution=resolution, up=1, activation=activation,
            resample_filter=None, use_noise=True)

        if self.torgb is not None:
            self.torgb = torgb_layer(oc_n, rgb_n, 1, w_dim=w_dim + w0_dim, activation=None)

    def forward(self, x, x0, img, ws, w0, fused_modconv=None, noise_mode='random'):
        dtype = torch.float16 if self.use_fp16 else torch.float32

        if fused_modconv is None:
            with misc.suppress_tracer_warnings():
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        x = x.to(dtype=dtype, memory_format=torch.contiguous_format)
        x0 = x0.to(dtype=dtype, memory_format=torch.contiguous_format)

        w_iter = iter(ws.unbind(dim=1))

        if self.res_link:
            y = self.skip(x, gain=np.sqrt(0.5))
            w_long = torch.cat([next(w_iter), w0], dim=1)
            x = self.conv0(x, w_long, fused_modconv=fused_modconv, noise_mode=noise_mode)
            x = x + x0
            w_long = torch.cat([next(w_iter), w0], dim=1)
            x = self.conv1(x, w_long, fused_modconv=fused_modconv, gain=np.sqrt(0.5), noise_mode=noise_mode)
            x = y.add_(x)
        else:
            w_long = torch.cat([next(w_iter), w0], dim=1)
            x = self.conv0(x, w_long, fused_modconv=fused_modconv, noise_mode=noise_mode)
            x = x + x0
            w_long = torch.cat([next(w_iter), w0], dim=1)
            x = self.conv1(x, w_long, fused_modconv=fused_modconv, noise_mode=noise_mode)

        if img is not None:
            img = upfirdn2d.upsample2d(img, self.resample_filter)

        to_rgb_out = None
        if self.torgb is not None:
            w_long = torch.cat([next(w_iter), w0], dim=1)
            y = self.torgb(x, w_long, fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            to_rgb_out = y
            img = img.add_(y) if img is not None else y

        return x, img, to_rgb_out


@register('comodgan_synthesis', version)
class Synthesis(nn.Module):
    def __init__(self,
                 w_dim=512,
                 w0_dim=1024,
                 resolution=256,
                 rgb_n=3,
                 ch_base=32768,
                 ch_max=512,
                 use_fp16_after_res=None,
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

        if resolution == 256:
            self.num_ws = 14
        elif resolution == 512:
            self.num_ws = 16

        hidden_ch = min(ch_base // block_res[0], ch_max)
        self.b4 = synthesis_block_first(
            w0_dim, hidden_ch, w_dim, resolution=4,
            rgb_n=rgb_n, activation=activation)

        for resi, resj in zip(block_res[:-1], block_res[1:]):
            hidden_ch_i = min(ch_base // resi, ch_max)
            hidden_ch_j = min(ch_base // resj, ch_max)
            use_fp16 = False if use_fp16_after_res is None else (resj > use_fp16_after_res)
            block = synthesis_block(
                hidden_ch_i, hidden_ch_j,
                w_dim=w_dim,
                w0_dim=w0_dim,
                resolution=resj,
                rgb_n=rgb_n,
                resample_filter=resample_filter,
                activation=activation,
                res_link=False,
                use_fp16=use_fp16, )

            setattr(self, 'b{}'.format(resj), block)

    def forward(self, x, feats, ws, noise_mode='random', return_intermediate_outs=False):
        block_ws = []
        ws = ws.to(torch.float32)
        w_idx = 0
        for res in self.block_res:
            block = getattr(self, f'b{res}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv

        w0 = x
        x, img = self.b4(x, feats[4], block_ws[0], noise_mode=noise_mode)

        intermediate_outputs = {"res_to_rgb": {4: img}, "res_img": {4: img}}
        for res, cur_ws in zip(self.block_res[1:], block_ws[1:]):
            block = getattr(self, f'b{res}')
            x, img, to_rgb_out = block(x, feats[res], img, cur_ws, w0, noise_mode=noise_mode)

            intermediate_outputs["res_to_rgb"][res] = to_rgb_out
            intermediate_outputs["res_img"][res] = img

        if return_intermediate_outs:
            return img, intermediate_outputs

        return img


@register('comodgan_generator', version)
class Generator(Generator_StyleGan):
    def __init__(self,
                 mapping,
                 encoder,
                 synthesis, ):

        super().__init__(mapping, synthesis)
        if isinstance(encoder, nn.Module):
            self.encoder = encoder
        else:
            self.encoder = get_model()(encoder)
        self.ic_n = self.encoder.ic_n

    def forward(self, x, z=None, c=None, truncation_psi=1, truncation_cutoff=None, noise_mode='random',
                return_intermediate_outs=False):
        """
        Args:
            x: 4 channel rgb+mask
        """
        if z is None:
            z = torch.randn([x.shape[0], 512]).to(x.device)
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        x, feats = self.encoder(x)

        if return_intermediate_outs:
            img, intermediate_outputs = self.synthesis(
                x, feats, ws, noise_mode=noise_mode, return_intermediate_outs=True)
            return img, intermediate_outputs

        img = self.synthesis(x, feats, ws, noise_mode=noise_mode)
        return img


@register('comodgan_discriminator', version)
class Discriminator(Discriminator_StyleGan):
    pass
