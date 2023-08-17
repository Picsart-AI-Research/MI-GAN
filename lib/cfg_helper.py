import os
import os.path as osp
import shutil
import copy
import time
import pprint
import numpy as np
import torch
import matplotlib
import argparse
import yaml
from easydict import EasyDict as edict

from .model_zoo import get_model

############
# cfg_bank #
############


def cfg_solvef(cmd, root):
    if not isinstance(cmd, str):
        return cmd
    
    if cmd.find('SAME')==0:
        zoom = root
        p = cmd[len('SAME'):].strip('()').split('.')
        p = [pi.strip() for pi in p]
        for pi in p:
            try:
                pi = int(pi)
            except:
                pass

            try:
                zoom = zoom[pi]
            except:
                return cmd
        return cfg_solvef(zoom, root)

    if cmd.find('SEARCH') == 0:
        zoom = root
        p = cmd[len('SEARCH'):].strip('()').split('.')
        p = [pi.strip() for pi in p]
        find = True
        # Depth first search
        for pi in p:
            try:
                pi = int(pi)
            except:
                pass
            
            try:
                zoom = zoom[pi]
            except:
                find = False
                break

        if find:
            return cfg_solvef(zoom, root)
        else:
            if isinstance(root, dict):
                for ri in root:
                    rv = cfg_solvef(cmd, root[ri])
                    if rv != cmd:
                        return rv
            if isinstance(root, list):
                for ri in root:
                    rv = cfg_solvef(cmd, ri)
                    if rv != cmd:
                        return rv
            return cmd

    if cmd.find('MODEL') == 0:
        goto = cmd[len('MODEL'):].strip('()')
        return model_cfg_bank()(goto)

    if cmd.find('DATASET') == 0:
        goto = cmd[len('DATASET'):].strip('()')
        return dataset_cfg_bank()(goto)

    return cmd


def cfg_solve(cfg, cfg_root):
    # The function solve cfg element such that 
    #   all sorrogate input are settled.
    #   (i.e. SAME(***) ) 
    if isinstance(cfg, list):
        for i in range(len(cfg)):
            if isinstance(cfg[i], (list, dict)):
                cfg[i] = cfg_solve(cfg[i], cfg_root)
            else:
                cfg[i] = cfg_solvef(cfg[i], cfg_root)
    if isinstance(cfg, dict):
        for k in cfg:
            if isinstance(cfg[k], (list, dict)):
                cfg[k] = cfg_solve(cfg[k], cfg_root)
            else:
                cfg[k] = cfg_solvef(cfg[k], cfg_root)        
    return cfg


class model_cfg_bank(object):
    def __init__(self):
        self.cfg_dir = osp.join('configs', 'model')
        self.cfg_bank = edict()
    
    def __call__(self, name):
        if name not in self.cfg_bank:
            cfg_path = self.get_yaml_path(name)
            with open(cfg_path, 'r') as f:
                cfg_new = yaml.load(
                    f, Loader=yaml.FullLoader)
            cfg_new = edict(cfg_new)
            self.cfg_bank.update(cfg_new)

        cfg = self.cfg_bank[name]
        cfg.name = name
        if 'super_cfg' not in cfg:
            cfg = cfg_solve(cfg, cfg)
            self.cfg_bank[name] = cfg
            return copy.deepcopy(cfg)

        super_cfg = self.__call__(cfg.super_cfg)
        # unlike other field,
        # args will not be replaced but update.
        if 'args' in cfg:
            super_cfg.args.update(cfg.args)
            cfg.pop('args')
        super_cfg.update(cfg)
        super_cfg.pop('super_cfg')
        cfg = super_cfg
        try:
            delete_args = cfg.pop('delete_args')
        except:
            delete_args = []

        for dargs in delete_args:
            cfg.args.pop(dargs)

        cfg = cfg_solve(cfg, cfg)
        self.cfg_bank[name] = cfg
        return copy.deepcopy(cfg)

    def get_yaml_path(self, name):
        if name.find('migan') == 0:
            return osp.join(
                self.cfg_dir, 'migan.yaml')
        else:
            raise ValueError


class dataset_cfg_bank(object):
    def __init__(self):
        self.cfg_dir = osp.join('configs', 'dataset')
        self.cfg_bank = edict()

    def __call__(self, name):
        if name not in self.cfg_bank:
            cfg_path = self.get_yaml_path(name)
            with open(cfg_path, 'r') as f:
                cfg_new = yaml.load(
                    f, Loader=yaml.FullLoader)
            cfg_new = edict(cfg_new)
            self.cfg_bank.update(cfg_new)

        cfg = self.cfg_bank[name]
        cfg.name = name
        if cfg.super_cfg is None:
            cfg = cfg_solve(cfg, cfg)
            self.cfg_bank[name] = cfg
            return copy.deepcopy(cfg)

        super_cfg = self.__call__(cfg.super_cfg)
        super_cfg.update(cfg)
        cfg = super_cfg
        cfg.super_cfg = None
        try:
            delete = cfg.pop('delete')
        except:
            delete = []

        for dargs in delete:
            cfg.pop(dargs)

        cfg = cfg_solve(cfg, cfg)
        self.cfg_bank[name] = cfg
        return copy.deepcopy(cfg)

    def get_yaml_path(self, name):
        if name.find('places2')==0:
            return osp.join(
                self.cfg_dir, 'places2.yaml')
        elif name.find('ffhq')==0:
            return osp.join(
                self.cfg_dir, 'ffhq.yaml')
        elif name.find('celeba')==0:
            return osp.join(
                self.cfg_dir, 'celeba.yaml')
        else:
            raise ValueError


class experiment_cfg_bank(object):
    def __init__(self):
        self.cfg_dir = osp.join('configs', 'experiment')
        self.cfg_bank = edict()

    def __call__(self, name):
        if name not in self.cfg_bank:
            cfg_path = self.get_yaml_path(name)
            with open(cfg_path, 'r') as f:
                cfg = yaml.load(
                    f, Loader=yaml.FullLoader)
            cfg = edict(cfg)

        cfg = cfg_solve(cfg, cfg)
        cfg = cfg_solve(cfg, cfg) 
        # twice for SEARCH
        self.cfg_bank[name] = cfg
        return copy.deepcopy(cfg)

    def get_yaml_path(self, name):
        return osp.join(
            self.cfg_dir, name+'.yaml')

##############
# cfg_helper #
##############


def get_experiment_id():
    time.sleep(0.5)
    return int(time.time()*100)


def cfg_to_debug(cfg):
    istrain = 'train' in cfg
    haseval = 'eval' in cfg
    if istrain:
        cfgt = cfg.train
        cfgt.experiment_id = 999999999999
        cfgt.signature = []
        cfgt.save_init_model = False

        if haseval:
            cfg.eval.experiment_id = cfgt.experiment_id    
            cfg.eval.signature = []
        
        cfgt.batch_size = None
        cfgt.batch_size_per_gpu = 2
        cfgt.dataset_num_workers = None
        cfgt.dataset_num_workers_per_gpu = 0

    if haseval:
        cfgt = cfg.eval
        cfgt.eval_tag = 'debug'

        cfgt.batch_size = None
        cfgt.batch_size_per_gpu = 1
        cfgt.dataset_num_workers = None
        cfgt.dataset_num_workers_per_gpu = 0

    cfg.env.matplotlib_mode = 'TKAgg'
    return cfg


def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--experiment', type=str)
    # parser.add_argument('--stage', nargs='+', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--gpu', nargs='+', type=int)
    parser.add_argument('--port', type=int)
    parser.add_argument('--signature', nargs='+', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--trainonly', action='store_true', default=False)

    parser.add_argument('--model', type=str)
    parser.add_argument('--eval', type=int)
    parser.add_argument('--pretrained_tag', type=str, default='last')
    parser.add_argument('--config', type=str)

    parser.add_argument('--dscache', type=float)
    parser.add_argument('--demo'   , action='store_true', default=False)
    parser.add_argument('--eval_tag', type=str)
    parser.add_argument('--pick'   , nargs='+', type=str)

    parser.add_argument('--pretrained_pkl', type=str, default=None)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--resume_path', type=str)
    parser.add_argument('--resume_itern', type=int)

    args = parser.parse_args()

    # Special handling the resume
    if args.resume_path is not None:
        cfg = edict()
        cfg.env = edict()
        cfg.env.debug = args.debug
        cfg.env.resume_path = args.resume_path
        cfg.env.resume_itern = args.resume_itern
        return cfg

    cfg = experiment_cfg_bank()(args.experiment)

    if args.model is not None:
        cfg.model = model_cfg_bank()(args.model)

    if args.dataset is not None:
        dataset = dataset_cfg_bank()(args.dataset)
        if 'train' in cfg:
            cfg.train.dataset = dataset
        if 'eval' in cfg:
            cfg.eval.dataset = dataset

    if args.dscache is not None:
        if 'train' in cfg:
            cfg.train.dataset.cache_pct = args.dscache
        if 'eval' in cfg:
            cfg.eval.dataset.cache_pct = args.dscache

    if args.demo is not None:
        if 'eval' in cfg:
            cfg.eval.dataset.demo = args.demo

    if args.pick is not None:
        if 'eval' in cfg:
            cfg.eval.dataset.pick = args.pick

    if (args.model is not None) \
            or (args.model is not None):
        cfg = cfg_solve(cfg, cfg)

    cfg.env.debug = args.debug

    if args.gpu is not None:
        cfg.env.gpu_device = list(args.gpu)

    if args.port is not None:
        port = int(args.port)
        cfg.env.dist_url = 'tcp://127.0.0.1:{}'.format(port)

    if args.eval is not None:
        if 'train' in cfg:
            cfg.pop('train')
        if 'eval' in cfg:
            cfg.eval.experiment_id = args.eval
            cfg.model.pretrained = '{}_{}_{}.pth'.format(
                cfg.eval.experiment_id, 
                cfg.model.symbol,
                args.pretrained_tag
            )
        else:
            raise ValueError

    if args.trainonly:
        if 'eval' in cfg:
            cfg.pop('eval')

    if args.signature is not None:
        if 'train' in cfg:
            cfg.train.signature = args.signature
        else:
            raise ValueError

    if args.seed is not None:
        cfg.env.rnd_seed = args.seed

    if args.eval_tag is not None:
        if 'eval' in cfg:
            cfg.eval.eval_tag = args.eval_tag

    if args.pretrained_pkl is not None:
        if 'eval' in cfg:
            cfg.eval.pretrained_pkl = args.pretrained_pkl

    return cfg


def cfg_initiates(cfg):
    """
    Step1.1: find the GPU device and count
    Step1.2: set batchsize and batchsize per gpu
    Step1.3: set dataloader worker and worker per gpu
    Step1.4: construct the main code path.
    Step1.5: set torch version.
 
    Step2.1: construct signature. 
    Step2.2: initialize log dir. 
    Step2.3: initialize log file.   

    Step3.1: save code.
    Step3.2: set random seed
    Step3.3: set random tracking
    Step3.4: try set matplotlib mode.
    """
    e = cfg.env
    isresume = 'resume_path' in e
    istrain = 'train' in cfg
    haseval = 'eval' in cfg

    if isresume:
        with open(osp.join(e.resume_path, 'config.yaml'), 'r') as f:
            cfg_resume = yaml.load(f, Loader=yaml.FullLoader)
        cfg_resume = edict(cfg_resume)
        cfg_resume.env.update(cfg.env)
        cfg = cfg_resume
        log_file = cfg.train.log_file

        print('')
        print('##########')
        print('# resume #')
        print('##########')
        print('')
        with open(log_file, 'a') as f:
            print('', file=f)
            print('##########', file=f)
            print('# resume #', file=f)
            print('##########', file=f)
            print('', file=f)

        pprint.pprint(cfg)
        with open(log_file, 'a') as f:
            pprint.pprint(cfg, f)
        return cfg
        
    # step1.1
    if e.gpu_device != 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
            [str(gid) for gid in e.gpu_device]) 
        e.gpu_count = len(e.gpu_device)
    else:
        e.gpu_count = torch.cuda.device_count()
    gpu_n = e.gpu_count

    # step1.2 - 1.3
    def temp(x, xp): 
        # Only one of x and xp can be None. 
        if (x is None) and (xp is None):
            raise ValueError
        elif x is None:
            x = xp * gpu_n
        elif xp is None:
            xp = x // gpu_n
        if x != xp * gpu_n:
            raise ValueError
        return x, xp

    if istrain:
        c = cfg.train
        c.batch_size, c.batch_size_per_gpu = \
            temp(c.batch_size, c.batch_size_per_gpu)
        c.dataset_num_workers, c.dataset_num_workers_per_gpu = \
            temp(c.dataset_num_workers, c.dataset_num_workers_per_gpu)
    if haseval:
        c = cfg.eval
        c.batch_size, c.batch_size_per_gpu = \
            temp(c.batch_size, c.batch_size_per_gpu)
        c.dataset_num_workers, c.dataset_num_workers_per_gpu = \
            temp(c.dataset_num_workers, c.dataset_num_workers_per_gpu)

    # step1.4 -- construct the main code path.
    # not sure needed or not

    # step1.5
    e.torch_version = torch.__version__

    # step2.1
    if istrain and (not e.debug):
        c = cfg.train
        version = get_model().get_version(cfg.model.type)
        try:
            i = c.signature.index('--hide--')
            sig1, sig2 = c.signature[0:i], c.signature[i:]
            c.signature = \
                ['v'+str(version)] + \
                sig1 + \
                ['s'+str(cfg.env.rnd_seed)] + \
                sig2
        except:
            c.signature = \
                ['v'+str(version)] + \
                c.signature + \
                ['s'+str(cfg.env.rnd_seed)]

    # step2.2
    if istrain:
        c = cfg.train
        try:
            i = c.signature.index('--hide--')
            sig = c.signature[0:i]
        except:
            sig = c.signature

        log_dir = [
            e.log_root_dir, 
            '{}_{}'.format(cfg.model.symbol, c.dataset.symbol),
            '_'.join([str(c.experiment_id)] + sig) 
        ]
        log_dir = osp.join(*log_dir)
        log_file = osp.join(log_dir, 'train.log')
        if not osp.exists(log_file):
            os.makedirs(osp.dirname(log_file))
        c.log_dir = log_dir
        c.log_file = log_file

        if haseval:
            cfg.eval.log_dir = log_dir
            cfg.eval.log_file = log_file
    else:
        # no train
        if haseval:
            # TODO
            c = cfg.eval
            log_dir = [
                e.log_root_dir, 
                '{}_{}'.format(cfg.model.symbol, c.dataset.symbol),
            ]
            exp_dir = search_experiment_folder(
                osp.join(*log_dir), c.experiment_id)
            pth_path = cfg.model.pretrained
            pth_path = log_dir + [exp_dir, pth_path]
            cfg.model.pretrained = osp.join(*pth_path)

            log_dir += [exp_dir, c.eval_tag]
            log_dir = osp.join(*log_dir)
            log_file = osp.join(log_dir, 'eval.log')
            if not osp.exists(log_file):
                os.makedirs(osp.dirname(log_file))
            c.log_dir = log_dir
            c.log_file = log_file

    # step2.3 print log and save config
    pprint.pprint(cfg)
    with open(log_file, 'w') as f:
        pprint.pprint(cfg, f)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(edict_2_dict(cfg), f)

    # step3.1 code saving
    if istrain:
        save_code = cfg.train.save_code
    elif haseval:
        save_code = cfg.eval.save_code
    else:
        save_code = False

    if save_code:
        codedir = osp.join(log_dir, 'code')
        if osp.exists(codedir):
            shutil.rmtree(codedir)
        for d in ['configs', 'lib']:
            fromcodedir = d
            tocodedir = osp.join(codedir, d)
            shutil.copytree(
                fromcodedir, tocodedir, 
                ignore=shutil.ignore_patterns(
                    '*__pycache__*', '*build*'))
        for codei in ['main.py']:
            shutil.copy(codei, codedir)

    # step3.2
    seed = e.rnd_seed
    if seed is None:
        pass
    elif isinstance(seed, int):
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        raise ValueError
        
    # step3.3
    # no need

    # step3.4
    try:
        if e.MATPLOTLIB_MODE is not None:
            matplotlib.use(e.matplotlib_mode)
    except:
        pass

    return cfg


def edict_2_dict(x):
    if isinstance(x, dict):
        xnew = {}
        for k in x:
            xnew[k] = edict_2_dict(x[k])
        return xnew
    elif isinstance(x, list):
        xnew = []
        for i in range(len(x)):
            xnew.append( edict_2_dict(x[i]) )
        return xnew
    else:
        return x


def search_experiment_folder(root, exid):
    for fi in os.listdir(root):
        if not osp.isdir(osp.join(root, fi)):
            continue
        if int(fi.split('_')[0]) == exid:
            return fi
    return None
