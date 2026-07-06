import os
from time import time
import argparse
from pathlib import Path

from einops import rearrange, reduce, repeat
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
import skimage
import torch

from .utils import load_image
from .model_utils import eval_transforms
from .hipt_4k import HIPT_4K
from .utils import load_pickle, save_pickle, join
from .image import upscale, smoothen
from .visual import plot_embeddings
from .connected_components import get_largest_connected


def _resolve_hipt_checkpoint_paths(
    checkpoint_path=None,
    vit256_checkpoint_path=None,
    vit4k_checkpoint_path=None,
):
    """Resolve HIPT ViT-256 and ViT-4K checkpoint files.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path, optional
        Directory containing ``vit256_small_dino.pth`` and ``vit4k_xs_dino.pth``.
        For backward compatibility, this may also be omitted, in which case the
        default ``checkpoints/`` folder next to this file is used.
    vit256_checkpoint_path : str or pathlib.Path, optional
        Explicit path to ``vit256_small_dino.pth``.
    vit4k_checkpoint_path : str or pathlib.Path, optional
        Explicit path to ``vit4k_xs_dino.pth``.
    """
    if vit256_checkpoint_path is None or vit4k_checkpoint_path is None:
        if checkpoint_path is None:
            checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
        else:
            checkpoint_dir = Path(checkpoint_path).expanduser().resolve()

        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"HIPT checkpoint directory not found: {checkpoint_dir}. "
                "Download HIPT checkpoints or pass checkpoint_path."
            )
        if not checkpoint_dir.is_dir():
            raise ValueError(
                "For HIPT, checkpoint_path should usually be a directory containing "
                "vit256_small_dino.pth and vit4k_xs_dino.pth."
            )

        if vit256_checkpoint_path is None:
            vit256_checkpoint_path = checkpoint_dir / "vit256_small_dino.pth"
        if vit4k_checkpoint_path is None:
            vit4k_checkpoint_path = checkpoint_dir / "vit4k_xs_dino.pth"

    vit256_checkpoint_path = Path(vit256_checkpoint_path).expanduser().resolve()
    vit4k_checkpoint_path = Path(vit4k_checkpoint_path).expanduser().resolve()

    missing = [str(p) for p in [vit256_checkpoint_path, vit4k_checkpoint_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing HIPT checkpoint file(s):\n  "
            + "\n  ".join(missing)
            + "\nExpected files: vit256_small_dino.pth and vit4k_xs_dino.pth."
        )

    return str(vit256_checkpoint_path), str(vit4k_checkpoint_path)


def match_foregrounds(embs, largest_only=False):
    print('Matching foregrounds...')
    t0 = time()
    channels = np.concatenate(list(embs.values()))
    mask = np.isfinite(channels).all(0)
    if largest_only:
        mask = get_largest_connected(mask)
    for group, channels in embs.items():
        for chan in channels:
            chan[~mask] = np.nan
    print(int(time() - t0), 'sec')


def patchify(x, patch_size):
    shape_ori = np.array(x.shape[:2])
    shape_ext = ((shape_ori + patch_size - 1) // patch_size * patch_size)
    x = np.pad(
        x,
        ((0, shape_ext[0] - x.shape[0]), (0, shape_ext[1] - x.shape[1]), (0, 0)),
        mode='edge',
    )
    tiles_shape = np.array(x.shape[:2]) // patch_size
    tiles = []
    for i0 in range(tiles_shape[0]):
        a0 = i0 * patch_size
        b0 = a0 + patch_size
        for i1 in range(tiles_shape[1]):
            a1 = i1 * patch_size
            b1 = a1 + patch_size
            tiles.append(x[a0:b0, a1:b1])
    return tiles, dict(original=shape_ori, padded=shape_ext, tiles=tiles_shape)


def get_data(prefix):
    slide_file = f'{prefix}he.png'
    if not os.path.exists(slide_file):
        slide_file = f'{prefix}he.jpg'
    slide = load_image(slide_file)

    mask_file = f'{prefix}mask.png'
    mask = load_image(mask_file)
    mask = np.array(mask) > 0
    mask = mask[..., np.newaxis].astype(np.uint8) * 255
    slide = np.concatenate([slide, mask], -1)
    return slide


def get_embeddings_sub(model, x):
    mask = x[..., -1] > 0
    x = x[..., :-1]
    x = x.astype(np.float32) / 255.0
    x = eval_transforms()(x)
    x_cls, x_sub = model.forward_all256(x[None])
    x_cls = x_cls.cpu().detach().numpy()
    x_sub = x_sub.cpu().detach().numpy()
    x_cls = x_cls[0].transpose(1, 2, 0)
    x_sub = x_sub[0].transpose(1, 2, 3, 4, 0)
    m_sub = reduce(
        mask,
        '(h2 h3 h) (w2 w3 w) -> h2 w2 h3 w3',
        'max',
        h2=x_sub.shape[0],
        w2=x_sub.shape[1],
        h3=x_sub.shape[2],
        w3=x_sub.shape[3],
    )
    x_sub[~m_sub] = np.nan
    return x_cls, x_sub


def get_embeddings_cls(model, x):
    x = torch.tensor(x.transpose(2, 0, 1))
    with torch.no_grad():
        __, x_sub4k = model.forward_all4k(x[None])
    x_sub4k = x_sub4k.cpu().detach().numpy()
    return x_sub4k[0].transpose(1, 2, 0)


def get_embeddings(
    img,
    pretrained=True,
    device='cuda',
    checkpoint_path=None,
    vit256_checkpoint_path=None,
    vit4k_checkpoint_path=None,
):
    """Extract HIPT embeddings from one image region."""
    print('Extracting embeddings...')
    t0 = time()
    tile_size = 4096
    tiles, shapes = patchify(img, patch_size=tile_size)

    model256_path, model4k_path = None, None
    if pretrained:
        model256_path, model4k_path = _resolve_hipt_checkpoint_paths(
            checkpoint_path=checkpoint_path,
            vit256_checkpoint_path=vit256_checkpoint_path,
            vit4k_checkpoint_path=vit4k_checkpoint_path,
        )

    model = HIPT_4K(
        model256_path=model256_path,
        model4k_path=model4k_path,
        device256=device,
        device4k=device,
    )
    model.eval()

    patch_size = (256, 256)
    subpatch_size = (16, 16)
    n_subpatches = tuple(a // b for a, b in zip(patch_size, subpatch_size))

    emb_sub = []
    emb_mid = []
    for i in range(len(tiles)):
        if i % 10 == 0:
            print('tile', i, '/', len(tiles))
        x_mid, x_sub = get_embeddings_sub(model, tiles[i])
        emb_mid.append(x_mid)
        emb_sub.append(x_sub)
    del tiles
    torch.cuda.empty_cache()

    emb_mid = rearrange(
        emb_mid,
        '(h1 w1) h2 w2 k -> (h1 h2) (w1 w2) k',
        h1=shapes['tiles'][0],
        w1=shapes['tiles'][1],
    )

    emb_cls = get_embeddings_cls(model, emb_mid)
    del emb_mid, model
    torch.cuda.empty_cache()

    shape_orig = np.array(shapes['original']) // subpatch_size

    chans_sub = []
    for i in range(emb_sub[0].shape[-1]):
        chan = rearrange(
            np.array([e[..., i] for e in emb_sub]),
            '(h1 w1) h2 w2 h3 w3 -> (h1 h2 h3) (w1 w2 w3)',
            h1=shapes['tiles'][0],
            w1=shapes['tiles'][1],
        )
        chan = chan[:shape_orig[0], :shape_orig[1]]
        chans_sub.append(chan)
    del emb_sub

    mask = np.isfinite(chans_sub[0])
    chans_cls = []
    for i in range(emb_cls[0].shape[-1]):
        chan = repeat(
            np.array([e[..., i] for e in emb_cls]),
            'h12 w12 -> (h12 h3) (w12 w3)',
            h3=n_subpatches[0],
            w3=n_subpatches[1],
        )
        chan = chan[:shape_orig[0], :shape_orig[1]]
        chan[~mask] = np.nan
        chans_cls.append(chan)
    del emb_cls

    print(int(time() - t0), 'sec')
    return chans_cls, chans_sub


def get_embeddings_shift(
    img,
    pretrained=True,
    device='cuda',
    checkpoint_path=None,
    vit256_checkpoint_path=None,
    vit4k_checkpoint_path=None,
):
    factor = 16
    margin = 256
    stride = 64
    shape_emb = np.array(img.shape[:2]) // factor
    chans_cls = [np.zeros(shape_emb, dtype=np.float32) for __ in range(192)]
    chans_sub = [np.zeros(shape_emb, dtype=np.float32) for __ in range(384)]
    start_list = list(range(0, margin, stride))
    n_reps = len(start_list) ** 2

    for start in start_list:
        start0, start1 = start, start
        print(f'shift {start0}/{margin}, {start1}/{margin}')
        t0 = time()
        stop0, stop1 = -margin + start0, -margin + start1
        im = img[start0:stop0, start1:stop1]
        cls, sub = get_embeddings(
            im,
            pretrained=pretrained,
            device=device,
            checkpoint_path=checkpoint_path,
            vit256_checkpoint_path=vit256_checkpoint_path,
            vit4k_checkpoint_path=vit4k_checkpoint_path,
        )
        del im
        sta0, sta1 = start0 // factor, start1 // factor
        sto0, sto1 = stop0 // factor, stop1 // factor
        for i in range(len(chans_cls)):
            chans_cls[i][sta0:sto0, sta1:sto1] += cls[i]
        del cls
        for i in range(len(chans_sub)):
            chans_sub[i][sta0:sto0, sta1:sto1] += sub[i]
        del sub
        print(int(time() - t0), 'sec')

    mar = margin // factor
    for chan in chans_cls:
        chan /= n_reps
        chan[-mar:] = np.nan
        chan[:, -mar:] = np.nan
    for chan in chans_sub:
        chan /= n_reps
        chan[-mar:] = np.nan
        chan[:, -mar:] = np.nan
    return chans_cls, chans_sub


def get_latent(x, n_components, method='pca', pre_normalize=False, post_normalize=False):
    if n_components >= 1:
        n_components = int(n_components)
    isfin = np.isfinite(x).all(-1)
    if pre_normalize:
        x -= x[isfin].mean(0)
        x /= x[isfin].std(0)
    if method == 'pca':
        model = PCA(n_components=n_components)
    elif method == 'umap':
        model = UMAP(n_components=n_components, n_neighbors=20, min_dist=0.0, n_jobs=64, random_state=0, verbose=True)
    else:
        raise ValueError(f'Method `{method}` not recognized')
    print(x[isfin].shape)
    u = model.fit_transform(x[isfin])
    print('n_components:', u.shape[-1], '/', x.shape[-1])
    if method == 'pca':
        print('pve:', model.explained_variance_ratio_.sum())
    order = np.nanvar(u, axis=0).argsort()[::-1]
    u = u[:, order]
    if post_normalize:
        u -= u.mean(0)
        u /= u.std(0)
    z = np.full(isfin.shape + (u.shape[-1],), np.nan, dtype=np.float32)
    z[isfin] = u
    return z, model


def smoothen_embeddings(embs, size, method='cnn', groups=None, device='cuda'):
    if groups is None:
        groups = embs.keys()
    out = {}
    for grp, em in embs.items():
        if grp in groups:
            if isinstance(em, list):
                smoothened = [smoothen(c[..., np.newaxis], size, method, device=device)[..., 0] for c in em]
            else:
                smoothened = smoothen(em, size, method, device=device)
        else:
            smoothened = em
        out[grp] = smoothened
    return out


def reduce_dim(embs, n_components, method='pca', balance=False, groups=None):
    print(f'Reducing dimension of embeddings using {method}...')
    if groups is None:
        groups = embs.keys()
    embs_dict = {}
    models_dict = {}
    for grp, em in embs.items():
        if grp in groups:
            t0 = time()
            em, mod = get_latent(em, n_components=n_components, method=method)
        else:
            mod = None
        embs_dict[grp] = em
        models_dict[grp] = mod
        print('runtime:', int(time() - t0), 'sec')
    return embs_dict, models_dict


def adjust_weights(embs, weights=None):
    print('Adjusting weights...')
    t0 = time()
    if weights is None:
        weights = {grp: 1.0 for grp in embs.keys()}
    for grp in embs.keys():
        channels = embs[grp]
        wt = weights[grp]
        means = np.array([np.nanmean(chan) for chan in channels])
        std = np.sum([np.nanvar(chan) for chan in channels]) ** 0.5
        for chan, me in zip(channels, means):
            chan[:] -= me
            chan[:] /= std
            chan[:] *= wt ** 0.5
    print(int(time() - t0), 'sec')


def save_embeddings(x, outfile):
    print('Saving embeddings...')
    t0 = time()
    save_pickle(x, outfile)
    print(int(time() - t0), 'sec')
    print('Embeddings saved to', outfile)


def hipt_extract_features(
    sample,
    device="cuda",
    reduction_method=None,
    n_components=None,
    smoothen_method="cv",
    random_weights=False,
    no_shift=False,
    use_cache=True,
    checkpoint_path=None,
    vit256_checkpoint_path=None,
    vit4k_checkpoint_path=None,
):
    """Extract HIPT grid-level embeddings using pretrained checkpoint paths."""
    np.random.seed(0)
    torch.manual_seed(0)

    prefix = f"{sample}/"
    wsi = get_data(prefix=prefix)
    raw_embs_file = prefix + "embeddings-hist-raw.pickle"

    # Resolve checkpoints before running expensive extraction. If random_weights=True,
    # pretrained weights are intentionally disabled and no files are required.
    if not random_weights:
        vit256_checkpoint_path, vit4k_checkpoint_path = _resolve_hipt_checkpoint_paths(
            checkpoint_path=checkpoint_path,
            vit256_checkpoint_path=vit256_checkpoint_path,
            vit4k_checkpoint_path=vit4k_checkpoint_path,
        )

    if use_cache and os.path.exists(raw_embs_file):
        embs = load_pickle(raw_embs_file)
        print("Embeddings loaded from", raw_embs_file)
    else:
        if not no_shift:
            emb_cls, emb_sub = get_embeddings_shift(
                wsi,
                pretrained=(not random_weights),
                device=device,
                checkpoint_path=checkpoint_path,
                vit256_checkpoint_path=vit256_checkpoint_path,
                vit4k_checkpoint_path=vit4k_checkpoint_path,
            )
        else:
            emb_cls, emb_sub = get_embeddings(
                wsi,
                pretrained=(not random_weights),
                device=device,
                checkpoint_path=checkpoint_path,
                vit256_checkpoint_path=vit256_checkpoint_path,
                vit4k_checkpoint_path=vit4k_checkpoint_path,
            )
        embs = dict(cls=emb_cls, sub=emb_sub)
        save_embeddings(embs, raw_embs_file)

    embs["rgb"] = np.stack([
        reduce(
            wsi[..., i].astype(np.float16) / 255.0,
            "(h1 h) (w1 w) -> h1 w1",
            "mean",
            h=16,
            w=16,
        ).astype(np.float32)
        for i in range(3)
    ])
    del wsi

    if smoothen_method is not None:
        print("Smoothening embeddings...")
        t0 = time()
        embs = smoothen_embeddings(embs, size=16, groups=["cls"], method=smoothen_method, device=device)
        print("runtime:", int(time() - t0))

    if reduction_method is not None:
        embs, reducers = reduce_dim(embs, n_components=n_components, method=reduction_method, balance=False, groups=["cls", "sub"])
        save_pickle(reducers, prefix + "reducers.pickle")

    match_foregrounds(embs)
    adjust_weights(embs)
    save_embeddings(embs, prefix + "embeddings-hist.pickle")
    print("----------Finished extracting HIPT embeddings----------")

