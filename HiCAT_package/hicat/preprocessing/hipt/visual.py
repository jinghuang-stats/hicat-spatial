import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from .utils import save_image
from .image import get_disk_mask


def cmap_turbo_truncated(x):
    cmap = plt.get_cmap('turbo')
    x = x * 0.9 + 0.05
    y = np.array(cmap(x))[..., :3]
    return y


def cmap_tab20(x):
    cmap = plt.get_cmap('tab20')
    x = x % 20
    x = (x // 10) + (x % 10) * 2
    return cmap(x)


def get_cmap_tab_multi(n_colors, n_shades, paired=True):
    cmap_base = cmap_tab20
    n_base = 10
    assert n_colors <= n_base

    def cmap(x):
        isin = x >= 0
        x = x * isin

        # lightness
        i = x // n_colors
        if paired:
            is_odd = i % 2 == 1
            i[~is_odd] //= 2
            i[is_odd] = n_shades - 1 - (i[is_odd] - 1) // 2

        # hue
        j = x % n_colors
        colors = np.stack(
                [cmap_base(k)[..., :3] for k in [j, j+n_base]])

        # compute color from hue and lightness
        weights = 1 - i / max(1, n_shades - 1)
        weights = np.stack([weights, 1-weights])
        col = (weights[..., np.newaxis] * colors).sum(0)
        col[~isin] = 0
        return col

    return cmap


def get_cmap_discrete(n_colors, cmap_name):
    cmap_base = plt.get_cmap(cmap_name)

    def cmap(x):
        x = x / float(n_colors-1)
        return cmap_base(x)

    return cmap


def plot_labels(labels, outfile, cmap=None):
    if labels.ndim == 3:
        n_labels = labels[..., 0].max() + 1
        n_shades = labels[..., -1].max() + 1
        isin = (labels >= 0).all(-1)
        labels_uni = labels[..., 0].copy()
        labels_uni[isin] = (
                n_labels * labels[isin][..., -1]
                + labels[isin][..., 0])
        labels = labels_uni
    elif labels.ndim == 2:
        n_labels = labels.max() + 1
        n_shades = 1

    if cmap is None:
        if n_labels <= 10:
            cmap = 'tab10'
        else:
            cmap = 'turbo'

    if cmap == 'tab20':
        cmap = cmap_tab20
        image = cmap(labels)[..., :3]
    elif cmap == 'turbo':
        cmap = plt.get_cmap('turbo')
        image = cmap(labels / labels.max())[..., :3]
    elif cmap == 'multi':
        cmap = get_cmap_tab_multi(n_labels, n_shades)
        image = cmap(labels)[..., :3]
    else:
        cmap = plt.get_cmap(cmap)
        image = cmap(labels)[..., :3]

    mask_extra = labels < 0
    mask_background = (labels == labels.min()) * mask_extra
    image[mask_extra] = 0.2
    image[mask_background] = 0.0
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.save(outfile)
    print(outfile)


def plot_embeddings(embeddings, prefix, groups=None, same_color_scale=True):
    if groups is None:
        groups = embeddings.keys()
    cmap = plt.get_cmap('turbo')
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    for key in groups:
        emb = embeddings[key]
        mask = np.all([np.isfinite(channel) for channel in emb], 0)
        min_all = np.min([channel[mask].min() for channel in emb], 0)
        max_all = np.max([channel[mask].max() for channel in emb], 0)
        for i, channel in enumerate(emb):
            if same_color_scale:
                min_chan, max_chan = min_all, max_all
            else:
                min_chan = channel[mask].min()
                max_chan = channel[mask].max()
            image = (channel - min_chan) / (max_chan - min_chan)
            image = cmap(image)[..., :3]
            if not mask.all():
                image[~mask] = 0.0
            image = Image.fromarray((image * 255).astype(np.uint8))
            outfile = f'{prefix}{key}-{i:02d}.png'
            image.save(outfile)
            print(outfile)


def plot_spots(
        img, cnts, locs, radius, outfile, cmap='magma', weight=0.5,
        disk_mask=True):
    img = img.astype(np.float32)
    cnts = cnts.astype(np.float32)

    img -= np.nanmin(img)
    img /= np.nanmax(img) + 1e-12
    img *= 1 - weight

    cnts -= np.nanmin(cnts)
    cnts /= np.nanmax(cnts) + 1e-12

    cmap = plt.get_cmap(cmap)
    mask = None
    if disk_mask:
        mask = get_disk_mask(radius)
    for ij, ct in zip(locs, cnts):
        i, j = ij
        color = cmap(ct)[:3]
        patch = np.full((radius*2, radius*2, 3), color)
        if mask is not None:
            patch[~mask] = 0
        patch *= weight
        img[i-radius:i+radius, j-radius:j+radius] += patch
    img = (img * 255).astype(np.uint8)
    save_image(img, outfile)


def plot_label_masks(labels, prefix):
    labs_uniq = np.unique(labels)
    labs_uniq = labs_uniq[labs_uniq >= 0]
    for lab in labs_uniq:
        mask = labels == lab
        save_image(mask, f'{prefix}label{lab:03d}.png')


def plot_matrix(img, outfile):
    img = img.astype(np.float32)
    img -= np.nanmin(img)
    img /= np.nanmax(img) + 1e-12
    cmap = plt.get_cmap('turbo')
    img = cmap(img)[..., :3]
    save_image((img * 255).astype(np.uint8), outfile)
