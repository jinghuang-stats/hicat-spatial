import numpy as np
import cv2 as cv
import torch
from torch import nn
import skimage


def impute_missing(x, mask, radius=3, method='ns'):
    method_dict = {
            'telea': cv.INPAINT_TELEA,
            'ns': cv.INPAINT_NS}
    method = method_dict[method]
    channels = [x[..., i] for i in range(x.shape[-1])]
    mask = mask.astype(np.uint8)
    y = [cv.inpaint(c, mask, radius, method) for c in channels]
    y = np.stack(y, -1)
    return y


def smoothen(
        x, size, method='cv', mode='mean',
        fill_missing=False, device='cuda'):
    mask = np.isfinite(x).all(-1)
    x = impute_missing(x, ~mask)
    if method == 'gf':
        y = skimage.filters.gaussian(
                x, sigma=size, preserve_range=True, channel_axis=-1)
    elif method == 'cv':
        kernel = np.ones((size, size), np.float32) / size**2
        y = cv.filter2D(
                x, ddepth=-1, kernel=kernel, borderType=cv.BORDER_REFLECT)
        if y.ndim == 2:
            y = y[..., np.newaxis]
    elif method == 'cnn':
        assert isinstance(size, int)
        padding = size // 2
        size = size + 1

        pool_dict = {
                'mean': nn.AvgPool2d(
                    kernel_size=size, stride=1, padding=0),
                'max': nn.MaxPool2d(
                    kernel_size=size, stride=1, padding=0)}
        pool = pool_dict[mode]

        mod = nn.Sequential(
                nn.ReflectionPad2d(padding),
                pool)
        y = mod(torch.tensor(x, device=device).permute(2, 0, 1))
        y = y.permute(1, 2, 0)
        y = y.cpu().detach().numpy()
    else:
        raise ValueError(f'Method `{method}` not recognized')
    if not fill_missing:
        y[~mask] = np.nan
    return y


def upscale(x, target_shape):
    mask = np.isfinite(x).all(tuple(range(2, x.ndim)))
    x = impute_missing(x, ~mask, radius=3)
    # TODO: Consider using pytorch with cuda to speed up
    # order: 0 == nearest neighbor, 1 == bilinear, 3 == bicubic
    x = skimage.transform.resize(
            x, target_shape, order=3, preserve_range=True)
    mask = skimage.transform.resize(
            mask.astype(float), target_shape, order=3,
            preserve_range=True)
    mask = mask > 0.5
    x[~mask] = np.nan
    return x


def crop_image(img, extent):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        img = np.pad(img, pad, mode='edge')
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    return img


def get_disk_mask(radius):
    locs = np.meshgrid(
            np.arange(-radius, radius), np.arange(-radius, radius),
            indexing='ij')
    locs = np.stack(locs, -1)
    isin = (locs**2).sum(-1) <= radius**2
    return isin
