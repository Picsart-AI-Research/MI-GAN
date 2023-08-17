import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import itertools

########
# unit #
########


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


def str2value(v):
    v = v.strip()
    try:
        return int(v)
    except:
        pass
    try:
        return float(v)
    except:
        pass
    if v in ('True', 'true'):
        return True
    elif v in ('False', 'false'):
        return False
    else:
        return v


@singleton
class get_unit(object):
    def __init__(self):
        self.unit = {}
        self.register('none', None)

        # general convolution
        self.register('conv', nn.Conv2d)
        self.register('bn', nn.BatchNorm2d)
        self.register('relu', nn.ReLU)
        self.register('relu6', nn.ReLU6)
        self.register('lrelu', nn.LeakyReLU)
        self.register('dropout', nn.Dropout)
        self.register('dropout2d', nn.Dropout2d)

    def register(self, 
                 name, 
                 unitf,):

        self.unit[name] = unitf

    def __call__(self, name):
        if name is None:
            return None
        i = name.find('(')
        i = len(name) if i==-1 else i
        t = name[:i]
        f = self.unit[t]
        args = name[i:].strip('()')
        if len(args) == 0:
            args = {}
            return f
        else:
            args = args.split('=')
            args = [[','.join(i.split(',')[:-1]), i.split(',')[-1]] for i in args]
            args = list(itertools.chain.from_iterable(args))
            args = [i.strip() for i in args if len(i)>0]
            kwargs = {}
            for k, v in zip(args[::2], args[1::2]):
                if v[0]=='(' and v[-1]==')':
                    kwargs[k] = tuple([str2value(i) for i in v.strip('()').split(',')])
                elif v[0]=='[' and v[-1]==']':
                    kwargs[k] = [str2value(i) for i in v.strip('[]').split(',')]
                else:
                    kwargs[k] = str2value(v)
            return functools.partial(f, **kwargs)


def register(name):
    def wrapper(class_):
        get_unit().register(name, class_)
        return class_
    return wrapper


@register('lrelu_agc')
# class lrelu_agc(nn.Module):
class lrelu_agc(object):
    """
    The lrelu layer with alpha, gain and clamp
    """
    def __init__(self, alpha=0.1, gain=1, clamp=None):
        # super().__init__()
        self.alpha = alpha
        if gain == 'sqrt_2':
            self.gain = np.sqrt(2)
        else:
            self.gain = gain
        self.clamp = clamp
        self.repr = 'lrelu_agc(alpha={}, gain={}, clamp={})'.format(
            alpha, gain, clamp)

    # def forward(self, x, gain=1):
    def __call__(self, x, gain=1):
        x = F.leaky_relu(x, negative_slope=self.alpha, inplace=True)
        act_gain = self.gain * gain
        act_clamp = self.clamp * gain if self.clamp is not None else None
        if act_gain != 1:
            x = x * act_gain
        if act_clamp is not None:
            x = x.clamp(-act_clamp, act_clamp)
        return x

    def __repr__(self,):
        return self.repr

##########
# helper #
##########


def freeze(net):
    for m in net.modules():
        if isinstance(m, (
                nn.BatchNorm2d, 
                nn.SyncBatchNorm,)):
            # inplace_abn not supported
            m.eval()
    for pi in net.parameters():
        pi.requires_grad = False
    return net


def common_init(m):
    if isinstance(m, (
            nn.Conv2d, 
            nn.ConvTranspose2d,)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (
            nn.BatchNorm2d, 
            nn.SyncBatchNorm,)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    else:
        pass


def init_module(module):
    """
    Args:
        module: [nn.module] list or nn.module
            a list of module to be initialized.
    """
    if isinstance(module, (list, tuple)):
        module = list(module)
    else:
        module = [module]

    for mi in module:
        for mii in mi.modules():
            common_init(mii)


def get_total_param(net):
    return sum(p.numel() for p in net.parameters())


def get_total_param_sum(net):
    with torch.no_grad():
        s = sum(p.cpu().detach().numpy().sum().item() for p in net.parameters())
    return s 
