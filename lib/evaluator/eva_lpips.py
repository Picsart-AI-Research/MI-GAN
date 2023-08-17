import torch
import numpy as np
import lpips

from ..log_service import print_log

from .eva_base import base_evaluator, register


@register('lpips')
class lpips_evaluator(base_evaluator):
    def __init__(self, net='alex', cuda=False, **dummy):
        super().__init__()
        self.symbol = 'lpips'
        self.data_lpips = None
        self.data_fn = None
        self.cuda = cuda
        self.compute_f = lpips.LPIPS(net=net)
        if cuda:
            self.RANK = torch.distributed.get_rank()
            self.compute_f = self.compute_f.to(self.RANK)
            self.compute_f = torch.nn.parallel.DistributedDataParallel(
                self.compute_f, device_ids=[self.RANK],)

    def add_batch(self, 
                  pred, 
                  gt, 
                  fn=None,
                  **dummy):

        if (pred.shape[1] != 3) and (gt.shape[1] != 3):
            raise ValueError
        if (len(pred.shape) != 4) and (len(gt.shape) != 4):
            raise ValueError

        # pred and gt need to be between 0 and 1 
        #   and will be normalized to -1 to 1
        pred = (pred-0.5)*2
        gt = (gt-0.5)*2

        if isinstance(pred, np.ndarray):
            pred = torch.Tensor(pred).float()
        if isinstance(gt, np.ndarray):
            gt = torch.Tensor(gt).float()

        if self.cuda:
            pred = pred.to(self.RANK)
            gt = gt.to(self.RANK)

        with torch.no_grad():
            lpips = self.compute_f(pred, gt).flatten()
    
        lpips = lpips.to('cpu').detach().numpy().astype(float)
        sync_lpips_fn = self.sync([lpips, fn])
        lpips, fn = zip(*sync_lpips_fn)
        lpips = self.zipzap_arrange(lpips)
        fn = self.zipzap_arrange(fn)

        if self.data_fn is None:
            self.data_fn = fn
        else:
            self.data_fn += fn

        if self.data_lpips is None:
            self.data_lpips = lpips
        else:
            self.data_lpips = np.concatenate(
                [self.data_lpips, lpips], axis=0)

    def compute(self):
        lpips = self.data_lpips[0:self.sample_n]
        lpips = lpips.mean()
        self.final['lpips'] = lpips
        return lpips

    def one_line_summary(self):
        print_log('Evaluator lpips: {:.4f}'.format(
            self.final['lpips']))

    def clear_data(self):
        self.data_lpips = None
        self.data_fn = None
