from time import time
import argparse
import os

import numpy as np
import skimage
import cv2 as cv
from einops import reduce, repeat
from scipy.ndimage import uniform_filter

from .tissue_mask import compute_tissue_mask
from .utils import load_image, save_image
from .image import crop_image


# def get_initial_mask(img):
#     img = img / 255.0
#     mask, __ = cluster(
#             img.transpose(2, 0, 1), n_clusters=2, method='mbkm')
#     for __ in range(1):
#         mask = (mask > 0).astype(np.float32)
#         mask = cv.blur(mask, (100, 100))
#     mask = mask > 0
#     if img[mask].var() < img[~mask].var():
#         mask = ~mask
#     return mask


def get_mask(img):
    print('Computing tissue mask...')
    t0 = time()
    # mask_init = get_initial_mask(img)
    mask_init = None
    mask = compute_tissue_mask(
            img, size_threshold=0.1, initial_mask=mask_init,
            max_iter=10)
    print(int(time() - t0), 'sec')
    return mask


def get_extent(mask):
    extent = []
    for ax in range(mask.ndim):
        ma = mask.swapaxes(0, ax)
        ma = ma.reshape(ma.shape[0], -1)
        notempty = ma.any(1)
        start = notempty.argmax()
        stop = notempty.size - notempty[::-1].argmax()
        extent.append([start, stop])
    extent = np.array(extent)
    return extent


def adjust_margins(img, mask, pad=0, cut=False):
    if cut:
        print('Removing margins...')
        extent = get_extent(mask)
        # make size divisible by pad
        extent += [0-pad, pad*2] - (extent % pad)
    else:
        print('Padding margins...')
        extent = np.stack([[0, 0], mask.shape]).T
        # make size divisible by pad without changing coords
        extent[:, 1] += pad*2
        extent[:, 1] -= (extent[:, 1] - extent[:, 0]) % pad
    img = crop_image(img, extent)
    mask = crop_image(mask[..., np.newaxis], extent)[..., 0]
    return img, mask, extent


def resize_image(img, shape):
    mask = np.isfinite(img).all(-1)
    img[~mask] = 0.0
    img = skimage.transform.resize(img, shape)
    mask = skimage.transform.resize(mask.astype(float), shape)
    mask = mask > 0.5
    img[~mask] = np.nan
    return img


def mirror_array(arr, mask, max_length=np.inf):
    assert np.ndim(arr) == 1
    assert np.shape(arr) == np.shape(mask)

    if mask.any() and (not mask.all()):
        arr = arr.copy()
        mask = mask.copy()

        # get start and stop of mask
        indices = np.where(mask)[0]
        start = indices.min()
        stop = indices.max() + 1
        # assert mask[start:stop].all()

        # reflect lower segment
        len_lower = min(stop-start, start, max_length)
        arr[(start-len_lower):start] = arr[start:(start+len_lower)][::-1]
        mask[(start-len_lower):start] = mask[start:(start+len_lower)][::-1]

        # reflect upper segment
        len_upper = min(stop-start, len(arr)-stop, max_length)
        arr[stop:(stop+len_upper)] = arr[(stop-len_upper):stop][::-1]
        mask[stop:(stop+len_upper)] = mask[(stop-len_upper):stop][::-1]

    return arr, mask


def mirror_image(img, mask, max_length=np.inf):
    shape_orig = img.shape
    img = img.reshape(img.shape[0], -1).T
    mask = np.tile(mask[..., np.newaxis], shape_orig[-1])
    mask = mask.reshape(mask.shape[0], -1).T
    out = [mirror_array(im, ma, max_length) for im, ma in zip(img, mask)]
    img = [e[0] for e in out]
    mask = [e[1] for e in out]
    img = np.stack(img).T.reshape(shape_orig)
    mask = np.stack(mask).T.reshape(shape_orig)[..., 0]
    return img, mask


def impute_background(img, mask, stride=None):

    img_orig = img.copy()
    mask_orig = mask.copy()

    if stride is not None:
        print('Reducing resolution...')
        t0 = time()

        # shape_down = np.array(img.shape[:2]) // stride
        # img = resize_image(img, shape_down)
        # img = (img * 255).astype(np.uint8)
        # mask = resize_image(mask[..., np.newaxis].astype(float), shape_down)
        # mask = mask[..., 0] > 0.5

        img = reduce(
                img.astype(float), '(h0 h1) (w0 w1) c -> h0 w0 c', 'mean',
                h1=stride, w1=stride).astype(np.uint8)
        mask = reduce(
                mask, '(h0 h1) (w0 w1) -> h0 w0', 'max',
                h1=stride, w1=stride)

        print(int(time() - t0), 'sec')

    print('Reflecting lower and upper segments...')
    t0 = time()
    n_iter = 5
    for __ in range(n_iter):
        for __ in range(2):  # reflect vertically then horizontally
            img, mask = mirror_image(img, mask)
            img, mask = img.swapaxes(0, 1), mask.swapaxes(0, 1)
    print(int(time() - t0), 'sec')

    print('Filling holes...')
    t0 = time()
    mask_bg = ((~mask) * 255).astype(np.uint8)
    img = cv.inpaint(img, mask_bg, 3, cv.INPAINT_TELEA)
    print(int(time() - t0), 'sec')

    if stride is not None:
        print('Restoring resolution...')
        t0 = time()
        # img = resize_image(img, img_orig.shape[:2])
        # img = (img * 255).astype(np.uint8)
        img = repeat(img, 'h0 w0 c -> (h0 h1) (w0 w1) c', h1=stride, w1=stride)
        print(int(time() - t0), 'sec')

    print('Copying imputed pixels...')
    t0 = time()
    img_orig[~mask_orig] = img[~mask_orig]
    print(int(time() - t0), 'sec')

    return img_orig


def get_image_filename(prefix):
    file_exists = False
    for suffix in ['.jpg', '.png', '.tif']:
        filename = prefix + suffix
        if os.path.exists(filename):
            file_exists = True
            break
    if not file_exists:
        raise FileNotFoundError('Image not found')
    return filename


def shrink_mask(x, size):
    size = size * 2 - 1
    x = uniform_filter(x.astype(float), size=size)
    x = np.isclose(x, 1)
    return x


def remove_border(x):
    x = x.copy()
    x[0] = 0
    x[-1] = 0
    x[:, 0] = 0
    x[:, -1] = 0
    return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--cut-margins', action='store_true')
    parser.add_argument('--mask', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # load high-resolution WSI
    img_filename = get_image_filename(f'{args.prefix}he-raw')
    img = load_image(img_filename)

    # get tissue mask
    if args.mask is None:
        mask = get_mask(img)
        mask = remove_border(mask)
    else:
        mask = load_image(args.mask) > 0

    # remove unnecessary margins
    img, mask, __ = adjust_margins(
            img, mask, pad=256, cut=args.cut_margins)
    # keep background
    save_image(img, f'{args.prefix}he-withbg.jpg')
    # remove background
    img[~mask] = 0
    save_image(img, f'{args.prefix}he.jpg')
    # reduce foreground to leave sufficient padding
    mask = shrink_mask(mask, size=256)
    # save mask
    save_image(mask, f'{args.prefix}mask.png')

    # impute background
    impute = False
    if impute:
        print('Imputing background...')
        t0 = time()
        img = impute_background(img, mask, stride=4)
        print(int(time() - t0), 'sec')


def preprocess_image(
    sample=None,
    mask=None,
    pad_size: int=256,

):
    """
    Preprocess H&E image for HIPT feature extraction.

    Parameters
    ----------
    sample : str
        Sample name of the image.
    mask : np.ndarray or None
        Tissue mask. If None, the tissue mask is computed automatically.
    pad_size : int
        Padding size for image extraction.
    """

    if sample is None:
        raise ValueError('Sample name must be provided.')

    if not os.path.exists(sample):
        raise FileNotFoundError('Sample folder not found')

    img_filename = get_image_filename(f'{sample}/he-raw')
    image = load_image(img_filename)


    # get tissue mask
    if mask is None:
        mask = get_mask(image)
        mask = remove_border(mask)
    else:
        mask = mask > 0 # convert provided mask to a boolean mask

    # remove unnecessary margins
    image_with_bg, mask, _ = adjust_margins(
        image,
        mask,
        pad=pad_size
    )
    save_image(image_with_bg, f"{sample}/he-withbg.jpg") # may need change path

    # remove background
    image_no_bg = image_with_bg.copy()
    image_no_bg[~mask] = 0
    save_image(image_no_bg, f"{sample}/he.jpg")

    # reduce foreground to leave sufficient padding
    mask = shrink_mask(
        mask,
        size=pad_size,
    )
    save_image(mask, f"{sample}/mask.png")

    print('----------Finished preprocessing----------')

    #return image_no_bg, image_with_bg, mask



if __name__ == '__main__':
    main()
