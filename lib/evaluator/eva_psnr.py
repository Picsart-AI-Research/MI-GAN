import numpy as np

from ..log_service import print_log

from .eva_base import base_evaluator, register

@register('psnr')
class psnr_evaluator(base_evaluator):
    def __init__(self, 
                 for_dataset='div2k',
                 scale=2,
                 rgb_range=1,
                 ):

        super().__init__()
        self.symbol = 'psnr'
        self.for_dataset = for_dataset
        self.scale = scale
        self.rgb_range = rgb_range
        self.data_psnr = None
        self.data_fn = None

    def add_batch(self, 
                  pred, 
                  gt, 
                  fn=None,
                  **dummy):

        if (pred.shape[1] != 3) and (gt.shape[1] != 3):
            raise ValueError
        if (len(pred.shape) != 4) and (len(gt.shape) != 4):
            raise ValueError

        diff = (pred-gt) / self.rgb_range

        if self.for_dataset is None:
            valid = diff
        elif self.for_dataset == 'benchmark':
            shave = self.scale
            if diff.shape[1] > 1:
                gray_coeffs = np.array([65.738, 129.057, 25.064])/256
                convert = gray_coeffs[None, :, None, None]
                diff = (diff*convert).sum(1)
            valid = diff[:, shave:-shave, shave:-shave]
        elif self.for_dataset == 'div2k':
            shave = self.scale + 6
            valid = diff[:, :, shave:-shave, shave:-shave]
        else:
            raise NotImplementedError

        if len(valid.shape) == 3:
            mse = (valid**2).mean(axis=(1, 2))
        else:
            mse = (valid**2).mean(axis=(1, 2, 3))

        psnr = -10*np.log10(mse)
        sync_psnr_fn = self.sync([psnr, fn])
        psnr, fn = zip(*sync_psnr_fn)
        psnr = self.zipzap_arrange(psnr)
        fn = self.zipzap_arrange(fn)

        if self.data_fn is None:
            self.data_fn = fn
        else:
            self.data_fn += fn

        if self.data_psnr is None:
            self.data_psnr = psnr
        else:
            self.data_psnr = np.concatenate(
                [self.data_psnr, psnr], axis=0)

    def compute(self):
        psnr = self.data_psnr[0:self.sample_n]
        psnr = psnr.mean()
        self.final['psnr'] = psnr
        return psnr

    def one_line_summary(self):
        print_log('Evaluator psnr: {:.4f}%'.format(
            self.final['psnr']))

    def clear_data(self):
        self.data_psnr = None
        self.data_fn = None
