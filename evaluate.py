import numpy as np
import torch
from torch import nn
from torch import Tensor
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def batch_PSNR(im_true: Tensor, im_fake: Tensor, data_range: int = 255) -> float:
    im_true, im_fake = im_true.detach(), im_fake.detach()
    N, C, H, W = im_true.size()
    Itrue = im_true.clamp(0.0, data_range).resize_(N, C * H * W)
    Ifake = im_fake.clamp(0.0, data_range).resize_(N, C * H * W)

    mse = nn.MSELoss(reduction="none")
    err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
    psnr = 10.0 * torch.log((data_range**2) / err) / np.log(10.0)
    return torch.mean(psnr).item()


def new_psnr(
    img_true: Tensor,
    img_test: Tensor,
    scale: int = 2,
    data_range: int = 256,
    chop: bool = False,
) -> float:
    if isinstance(img_true, torch.Tensor):
        img_true, img_test = (
            img_true.detach().cpu().numpy(),
            img_test.detach().cpu().numpy(),
        )
    if len(img_true.shape) == 4:
        assert (
            img_true.shape[0] == 1
        ), "compute the PSNR for batch data please use batch_psnr"
        img_true, img_test = img_true.squeeze(0), img_test.squeeze(0)
    if chop:
        img_true, img_test = (
            img_true[:, scale:-scale, scale:-scale],
            img_test[:, scale:-scale, scale:-scale],
        )
    if img_true.shape[0] == 3 or img_true.shape[0] == 1:
        img_true, img_test = img_true.transpose([1, 2, 0]), img_test.transpose(
            [1, 2, 0]
        )
    return compare_psnr(img_true, img_test, data_range=data_range)


def new_ssim(
    img_true: Tensor,
    img_test: Tensor,
    scale: int = 2,
    data_range: int = 256,
    chop: bool = False,
) -> float:
    if isinstance(img_true, torch.Tensor):
        img_true, img_test = (
            img_true.detach().cpu().numpy(),
            img_test.detach().cpu().numpy(),
        )
    if len(img_true.shape) == 4:
        assert (
            img_true.shape[0] == 1
        ), "compute the SSIM for batch data please use batch_psnr"
        img_true, img_test = img_true.squeeze(0), img_test.squeeze(0)
    if chop:
        img_true, img_test = (
            img_true[:, scale:-scale, scale:-scale],
            img_test[:, scale:-scale, scale:-scale],
        )
    if img_true.shape[0] == 3:
        img_true, img_test = img_true.transpose([1, 2, 0]), img_test.transpose(
            [1, 2, 0]
        )
        return compare_ssim(
            img_true,
            img_test,
            win_size=11,
            data_range=data_range,
            gaussian_weights=True,
            multichannel=True,
        )
    elif img_true.shape[0] == 1:
        img_true, img_test = img_true[0], img_test[0]
        return compare_ssim(
            img_true,
            img_test,
            win_size=11,
            data_range=data_range,
            gaussian_weights=True,
        )
    else:
        return compare_ssim(
            img_true,
            img_test,
            win_size=11,
            data_range=data_range,
            gaussian_weights=True,
        )
        assert False
