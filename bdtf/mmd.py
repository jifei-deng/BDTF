import torch


def _gaussian(diff_sq_sum, sigma):
    return torch.exp(-diff_sq_sum / (2.0 * sigma))


def _laplacian(abs_sum, sigma):
    return torch.exp(-abs_sum / (2.0 * sigma))


def mmd(x, y, kernel="gaussian", sigma=10.0, squared=True):
    if kernel == "gaussian":
        xx = _gaussian(((x.unsqueeze(2) - x.unsqueeze(1)) ** 2).sum(-1), sigma)
        xy = _gaussian(((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1), sigma)
        yy = _gaussian(((y.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1), sigma)
    elif kernel == "laplacian":
        xx = _laplacian((x.unsqueeze(2) - x.unsqueeze(1)).abs().sum(-1), sigma)
        xy = _laplacian((x.unsqueeze(2) - y.unsqueeze(1)).abs().sum(-1), sigma)
        yy = _laplacian((y.unsqueeze(2) - y.unsqueeze(1)).abs().sum(-1), sigma)
    else:
        raise ValueError(f"unknown kernel: {kernel}")
    out = (xx.mean(dim=(1, 2)) - 2.0 * xy.mean(dim=(1, 2)) + yy.mean(dim=(1, 2))).clamp(min=0.0)
    return out if squared else (out + 1e-6).sqrt()
