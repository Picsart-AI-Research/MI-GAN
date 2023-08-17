# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils


class compute_fid_inpainting(object):
    def __init__(self, opts, max_real=None, num_gen=50000):
        self.compute_ds  = metric_utils.compute_feature_stats_for_dataset
        self.compute_gen = metric_utils.compute_feature_stats_for_inpainting
        # Direct TorchScript translation of
        # http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
        self.detector_url = \
            'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
        self.detector_kwargs = {
            'return_features' : True
        }
        self.opts = opts
        self.max_real = max_real
        self.num_gen = num_gen

    def __call__(self):
        mu_real, sigma_real = self.compute_ds(
            opts=self.opts,
            detector_url=self.detector_url,
            detector_kwargs=self.detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True,
            max_items=self.max_real, ).get_mean_cov()

        mu_gen, sigma_gen = self.compute_gen(
            opts=self.opts,
            detector_url=self.detector_url,
            detector_kwargs=self.detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True,
            max_items=self.num_gen).get_mean_cov()

        if self.opts.rank != 0:
            return float('nan')

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)
