import torch
import torch.distributed as dist

import numpy as np
import timeit

from .cfg_holder import cfg_unique_holder as cfguh
from .log_service import print_log


class exec_container(object):
    """
    This is the base functor for all types of executions.
        One execution can have multiple stages, 
        but are only allowed to use the same 
        config, network, dataloader. 
    Thus, in most of the cases, one exec_container is one
        training/evaluation/demo...
    If DPP is in use, this functor should be spawn.
    """
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.registered_stages = []
        self.RANK = None

    def register_stage(self, stage):
        self.registered_stages.append(stage)

    def __call__(self, RANK, **kwargs):
        """
        Args:
            RANK: int,
                the rank of the stage process.
            If not multi process, please set 0.
        """
        self.RANK = RANK
        cfg = self.cfg
        cfguh().save_cfg(cfg)

        # broadcast cfg
        dist.init_process_group(
            backend=cfg.env.dist_backend,
            init_method=cfg.env.dist_url,
            rank=RANK,
            world_size=cfg.env.gpu_count,
        )

        # need to set random seed 
        # originally it is in common_init()
        # but it looks like that the seed doesn't transfered to here.
        if isinstance(cfg.env.rnd_seed, int):
            np.random.seed(cfg.env.rnd_seed)
            torch.manual_seed(cfg.env.rnd_seed)

        time_start = timeit.default_timer()

        para = {'RANK': RANK, 'itern_total': 0}

        for stage in self.registered_stages:
            stage_para = stage(**para)
            if stage_para is not None:
                para.update(stage_para)

        print_log('Total {:.2f} seconds'.format(timeit.default_timer() - time_start))
        self.RANK = None
        dist.destroy_process_group()
