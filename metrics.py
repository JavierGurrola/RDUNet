import torch
from pytorch_msssim import SSIM as _SSIM


class PSNR(object):
    r"""
    Evaluates the PSNR metric in a tensor.
    It can return a result with different reduction methods.

    Args:
        data_range (int, float): Range of the input images.
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
        eps (float): Epsilon value to avoid division by zero.
    """
    def __init__(self, data_range, reduction='none', eps=1e-8):
        self.data_range = data_range
        self.reduction = reduction
        self.eps = eps

    def __call__(self, outputs, targets):
        with torch.set_grad_enabled(False):
            mse = torch.mean((outputs - targets) ** 2., dim=(1, 2, 3))
            psnr = 10. * torch.log10((self.data_range ** 2.) / (mse + self.eps))

            if self.reduction == 'mean':
                return psnr.mean()
            if self.reduction == 'sum':
                return psnr.sum()

            return psnr


class SSIM(object):
    r"""
    Evaluates the SSIM metric in a tensor.
    It can return a result with different reduction methods.

    Args:
        channels (int): Number of channels of the images.
        data_range (int, float): Range of the input images.
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
    """
    def __init__(self, channels, data_range, reduction='none'):
        self.data_range = data_range
        self.reduction = reduction
        self.ssim_module = _SSIM(data_range=data_range, size_average=False, channel=channels)

    def __call__(self, outputs, targets):
        with torch.set_grad_enabled(False):
            ssim = self.ssim_module(outputs, targets)

            if self.reduction == 'mean':
                return ssim.mean()
            if self.reduction == 'sum':
                return ssim.sum()

            return ssim
