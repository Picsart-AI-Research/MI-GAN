import copy
import itertools

import torch

from .ds_loader import get_loader
from .ds_formatter import get_formatter
from ...log_service import print_log


class ds_base(torch.utils.data.Dataset):
    def __init__(self, cfg, loader=None, formatter=None):
        self.cfg = cfg
        self.init_load_info(cfg)
        self.loader = loader
        self.formatter = formatter

        console_info = '{}: '.format(self.__class__.__name__)
        console_info += 'total {} unique images, '.format(len(self.load_info))

        for idx, info in enumerate(self.load_info):
            info['idx'] = idx
        try:
            self.repeat = self.cfg.repeat
        except:
            self.repeat = 1

        console_info += 'total {} unique sample. Repeat {} times.'.format(len(self.load_info), self.repeat)
        print_log(console_info)

    def init_load_info(self, cfg):
        # implement by sub class
        raise ValueError

    def __len__(self):
        return len(self.load_info) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.load_info)

        element = self.load_info[idx]
        element = copy.deepcopy(element)
        element = self.loader(element)

        if self.formatter is not None:
            return self.formatter(element)
        else:
            return element


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


@singleton
class get_dataset(object):
    def __init__(self):
        self.dataset = {}

    def register(self, ds):
        self.dataset[ds.__name__] = ds

    def __call__(self, cfg):
        t = cfg.type

        # the register is in each file
        if t == 'places2':
            from .. import ds_places2
        elif t in ['ffhq', 'ffhqsimple', 'ffhqzip']:
            from .. import ds_ffhq
        else:
            raise ValueError

        loader = get_loader()(cfg.loader)
        formatter = get_formatter()(cfg.formatter)

        return self.dataset[t](cfg, loader, formatter)


def register():
    def wrapper(class_):
        get_dataset().register(class_)
        return class_
    return wrapper

# some other helpers


class collate(object):
    """
        Modified from torch.utils.data._utils.collate
        It handle list different from the default.
            List collate just by append each other.
    """
    def __init__(self):
        self.default_collate = \
            torch.utils.data._utils.collate.default_collate

    def __call__(self, batch):
        """
        Args:
            batch: [data, data] -or- [(data1, data2, ...), (data1, data2, ...)]
        This function will not be used as induction function
        """
        elem = batch[0]
        if not (elem, (tuple, list)):
            return self.default_collate(batch)
        
        rv = []
        # transposed
        for i in zip(*batch):
            if isinstance(i[0], list):
                if len(i[0]) != 1:
                    raise ValueError
                try:
                    i = [[self.default_collate(ii).squeeze(0)] for ii in i]
                except:
                    pass
                rvi = list(itertools.chain.from_iterable(i))
                rv.append(rvi) # list concat
            else:
                rv.append(self.default_collate(i))
        return rv
