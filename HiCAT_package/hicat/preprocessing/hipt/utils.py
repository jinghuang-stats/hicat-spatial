import itertools
from PIL import Image
import pickle
import os

import numpy as np
import pandas as pd


Image.MAX_IMAGE_PIXELS = None


def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def load_image(filename):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    print(f'Image loaded from {filename}')
    return img


def save_image(img, filename):
    mkdir(filename)
    Image.fromarray(img).save(filename)
    print(filename)


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def write_lines(strings, filename):
    mkdir(filename)
    with open(filename, 'w') as file:
        for s in strings:
            file.write(f'{s}\n')
    print(filename)


def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    print(f'Pickle loaded from {filename}')
    return x


def load_tsv(infile):
    return pd.read_csv(infile, sep='\t', header=0, index_col=0)


def save_tsv(x, filename):
    mkdir(filename)
    x.to_csv(filename, sep='\t')
    print(filename)


def join(x):
    return list(itertools.chain.from_iterable(x))


def get_most_frequent(x):
    # return the most frequent element in array
    uniqs, counts = np.unique(x, return_counts=True)
    return uniqs[counts.argmax()]


def sort_labels(labels, descending=True):
    labels = labels.copy()
    isin = labels >= 0
    __, labels[isin], counts = np.unique(
            labels[isin], return_inverse=True, return_counts=True)
    c = counts
    if descending:
        c = c * (-1)
    order = c.argsort()
    rank = order.argsort()
    labels[isin] = rank[labels[isin]]
    return labels, order
