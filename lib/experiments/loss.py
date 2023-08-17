# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):  # to be overridden by subclass
        raise NotImplementedError()


class MIGANLoss(Loss):
    def __init__(
        self,
        device,
        G_synthesis,
        D,
        r1_gamma=10,
        G_encoder=None,
        image_level_kd_kwargs=None
    ):
        super().__init__()
        self.G_encoder = G_encoder
        self.device = device
        self.G_synthesis = G_synthesis
        self.D = D
        self.r1_gamma = r1_gamma

        self.teacher1_G = None
        self.teacher2_G = None

        self.image_level_kd_kwargs = image_level_kd_kwargs

        if image_level_kd_kwargs is not None:
            self.use_image_level_kd = image_level_kd_kwargs["use_image_level_kd"]
            if self.use_image_level_kd:
                self.load_teacher1_model(
                    image_level_kd_kwargs["teacher1_path"],
                    device=device,
                    resolution=image_level_kd_kwargs["inference_resolution"]
                )

    def load_teacher1_model(self, path, device, resolution=256, load_D=False):
        if self.teacher1_G is not None:
            return

        path = Path(path)
        print('Loading teacher 1 (CoModGAN) model from {}...'.format(path))
        if path.suffix == ".pkl":
            with open(path, 'rb') as f:
                teacher_data = pickle.Unpickler(f).load()
            if load_D:
                self.teacher1_D = teacher_data['D'].to(device)
            self.teacher1_G = teacher_data['G_ema'].to(device)
            self.teacher1_G.eval()
        elif path.suffix == ".pth" or path.suffix == ".pt":
            from lib.model_zoo.comodgan import Mapping, Encoder, Synthesis, Generator

            setting = {
                'mapping': {
                    'activation': 'lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
                    'c_dim': 0,
                    'embed_features': None,
                    'layer_features': None,
                    'lr_multiplier': 0.01,
                    'num_layers': 8,
                    'num_ws': 14 if resolution == 256 else 16,
                    'w_avg_beta': 0.995,
                    'w_dim': 512,
                    'z_dim': 512,
                },
                'encoder': {
                    'activation': 'lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
                    'c_dim': None,
                    'ch_base': 32768,
                    'ch_max': 512,
                    'cmap_dim': None,
                    'has_extra_final_layer': False,
                    'ic_n': 4,
                    'mbstd_c_n': 0,
                    'mbstd_group_size': 0,
                    'oc_n': 1024,
                    'resample_filter': [1, 3, 3, 1],
                    'resolution': resolution,
                    'use_dropout': True,
                    'use_fp16_before_res': None,
                },
                'synthesis': {
                    'activation': 'lrelu_agc(alpha=0.2, gain=sqrt_2, clamp=256)',
                    'ch_base': 32768,
                    'ch_max': 512,
                    'resample_filter': [1, 3, 3, 1],
                    'resolution': resolution,
                    'rgb_n': 3,
                    'use_fp16_after_res': None,
                    'w0_dim': 1024,
                    'w_dim': 512,
                }
            }

            mapping = Mapping(**setting['mapping'])
            encoder = Encoder(**setting['encoder'])
            synthesis = Synthesis(**setting['synthesis'])
            G = Generator(mapping, encoder, synthesis)
            G.load_state_dict(torch.load(path))
            if load_D:
                raise Exception("Can't load discriminator from exported generator pt.")
            self.teacher1_G = G.to(device)
            self.teacher1_G.eval()

    def run_G(self, x, sync):
        with misc.ddp_sync(self.G_encoder, sync):
            x_global, feats = self.G_encoder(x)

        with misc.ddp_sync(self.G_synthesis, sync):
            img, intermediate_outputs = self.G_synthesis(x_global, feats)
        return img, intermediate_outputs

    def run_teacher1_G(self, x, sync):
        z = torch.randn([x.shape[0], self.teacher1_G.z_dim], device=x.device)
        c = torch.randn([x.shape[0], self.teacher1_G.c_dim], device=x.device)
        with torch.no_grad() and misc.ddp_sync(self.teacher1_G, sync):
            img, intermediate_outputs = self.teacher1_G(x, z, c, return_intermediate_outs=True)

        return img, intermediate_outputs

    def run_D(self, img, sync):
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img)
        return logits

    def accumulate_gradients(
        self,
        phase,
        real_img,
        mask,
        real_img_erased,
        sync,
        gain
    ):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_x = torch.cat([mask - 0.5, real_img_erased], dim=1)
                gen_img_full, gen_intermediate_outs = self.run_G(gen_x, sync=sync)  # May get synced by Gpl.
                gen_img_combined = gen_img_full * (1 - mask) + real_img * mask
                gen_img = torch.cat([mask - 0.5, gen_img_combined], dim=1)
                gen_logits = self.run_D(gen_img, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))

                # Teacher 1 KD, image level
                if self.image_level_kd_kwargs is not None and self.use_image_level_kd:
                    kd_l1_image_level_loss = 0
                    t1_img_full, t1_intermediate_outs = self.run_teacher1_G(gen_x, sync=sync)
                    for res in t1_intermediate_outs["res_to_rgb"].keys():
                        if res >= self.image_level_kd_kwargs["start_resolution"]:
                            gen_res_to_rgb = gen_intermediate_outs["res_to_rgb"][res]
                            t1_res_to_rgb = t1_intermediate_outs["res_to_rgb"][res]
                            resized_mask = F.interpolate(mask, size=gen_res_to_rgb.shape[2:], mode='nearest').detach()

                            kd_l1_image_level_loss += torch.mean(
                                F.l1_loss(gen_res_to_rgb, t1_res_to_rgb.detach(), reduction='none') * (1 - resized_mask)
                            )

                    training_stats.report('Loss/G/kd_l1_image_level_loss', kd_l1_image_level_loss)

                    loss_Gmain = loss_Gmain + self.image_level_kd_kwargs["weight"] * kd_l1_image_level_loss

                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_x = torch.cat([mask - 0.5, real_img_erased], dim=1)
                gen_img_full, _ = self.run_G(gen_x, sync=False)
                gen_img_combined = gen_img_full * (1 - mask) + real_img * mask
                gen_img = torch.cat([mask - 0.5, gen_img_combined], dim=1)
                gen_logits = self.run_D(gen_img, sync=False)  # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_x = torch.cat([mask - 0.5, real_img], dim=1).detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_x, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_x], create_graph=True,
                                                       only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
