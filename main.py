import torch.multiprocessing as mp

import lib.model_zoo
from lib.cfg_helper import \
    get_experiment_id, \
    get_command_line_args, \
    cfg_initiates

from lib.utils import exec_container
from lib.experiments import get_experiment


if __name__ == "__main__":
    cfg = get_command_line_args()
    isresume = 'resume_path' in cfg.env

    if 'train' in cfg and not isresume:
        cfg.train.experiment_id = get_experiment_id()

    cfg = cfg_initiates(cfg)

    if 'train' in cfg: 
        trainer = exec_container(cfg)
        tstage = get_experiment(cfg.train.exec_stage)()
        trainer.register_stage(tstage)

        mp.spawn(trainer, args=(), nprocs=cfg.env.gpu_count, join=True)
    else:
        evaler = exec_container(cfg)
        estage = get_experiment(cfg.eval.exec_stage)()
        evaler.register_stage(estage)
        if cfg.env.debug:
            evaler(0)
        else:
            mp.spawn(evaler, args=(), nprocs=cfg.env.gpu_count, join=True)
