import torch
import torch.nn.functional as F
import numpy as np

from ..log_service import print_log

from .eva_base import base_evaluator, register

# Reference: https://github.com/Po-Hsun-Su/pytorch-ssim

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.FloatTensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def compute_ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

@register('ssim')
class ssim_evaluator(base_evaluator):
    def __init__(self, 
                 window_size=11,):

        super().__init__()
        self.symbol = 'ssim'
        self.window_size = window_size
        self.data_ssim = None
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

        pred = torch.FloatTensor(pred)
        gt = torch.FloatTensor(gt)
        ssim = compute_ssim(pred, gt, window_size=self.window_size, size_average=False)
        ssim = ssim.detach().to('cpu').numpy().astype(float)

        sync_ssim_fn = self.sync([ssim, fn])
        ssim, fn = zip(*sync_ssim_fn)
        ssim = self.zipzap_arrange(ssim)
        fn = self.zipzap_arrange(fn)

        for ssim, fn in sync_ssim_fn:
            self.data_fn = fn if self.data_fn is None else self.data_fn + fn
            self.data_ssim = ssim if self.data_ssim is None else np.concatenate([self.data_ssim, ssim], axis=0)

    def compute(self):
        ssim = self.data_ssim[0:self.sample_n]
        ssim = ssim.mean()
        self.final['ssim'] = ssim
        return ssim

    def one_line_summary(self):
        print_log('Evaluator ssim: {:.4f}'.format(
            self.final['ssim']))

    def clear_data(self):
        self.data_ssim = None
        self.data_fn = None
