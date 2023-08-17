
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import json
import torch
import dnnlib

from . import metric_utils
from . import frechet_inception_distance


#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

#----------------------------------------------------------------------------

def calc_metric(metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    
    # XX_hack
    evalloader = kwargs.pop('evalloader', None)

    opts = metric_utils.MetricOptions(**kwargs)

    # XX_hack
    setattr(opts, 'evalloader', evalloader)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    
    metric_name = metric
    if 'metric_name' in results:
        metric_name = results['metric_name']
        del results['metric_name']

    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric_name,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------

def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------
# Primary metrics.

@register_metric
def fid50k_full_inpainting(opts):
    fid = frechet_inception_distance.compute_fid_inpainting(opts, max_real=None, num_gen=50000)
    return {'fid50k_full_inpainting' : fid()}

@register_metric
def fid10k_full_inpainting(opts):
    fid = frechet_inception_distance.compute_fid_inpainting(opts, max_real=None, num_gen=10000)
    return {'fid10k_full_inpainting' : fid()}

@register_metric
def fid36500_full_inpainting(opts):
    fid = frechet_inception_distance.compute_fid_inpainting(opts, max_real=None, num_gen=36500)
    return {'fid36500_full_inpainting' : fid()}

#----------------------------------------------------------------------------

#############
# Metrics for CelebA experiments
#############

@register_metric
def fid5k_full_inpainting(opts):
    fid = frechet_inception_distance.compute_fid_inpainting(opts, max_real=None, num_gen=5000)
    return {'fid5k_full_inpainting' : fid()}
