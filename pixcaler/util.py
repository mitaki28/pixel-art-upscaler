import random
import math
import itertools
from pathlib import Path
from fractions import Fraction

import numpy as np
from PIL import Image

def img_to_chw_array(img):
    return np.asarray(img.convert('RGBA')).astype("f").transpose((2, 0, 1)) / 127.5 - 1.0

def chw_array_to_img(x):
    C, H, W = x.shape
    x = x.transpose(1, 2, 0)
    if C==1:
        x = x.reshape((H, W))
    else:
        x = x.reshape((H, W, C))
    return Image.fromarray(
        np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    )

def img_to_hwc_array(img):
    return np.asarray(img.convert('RGBA')).astype("f") / 127.5 - 1.0

def hwc_array_to_img(x):
    H, W, C = x.shape
    if C==1:
        x = x.reshape((H, W))
    else:
        x = x.reshape((H, W, C))
    return Image.fromarray(
        np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    )

def upsample_nearest_neighbor(img, r):
    c, h, w = img.shape
    f = Fraction(r)
    assert h % f.denominator == 0 and w % f.denominator == 0
    nh, nw = h // f.denominator * f.numerator, w // f.denominator * f.numerator
    idx = np.indices((nh, nw))
    idx *= f.denominator
    idx //= f.numerator
    return img[:,idx[0],idx[1]]

def downsample_nearest_neighbor(img, r):
    return upsample_nearest_neighbor(img, Fraction(1) / Fraction(r))

def downsample_random_nearest_neighbor(img, r):
    c, h, w = img.shape
    nh, nw = int(h // r), int(w // r)
    idx = np.indices((nh, nw))
    idx = (r * (idx + np.random.random(idx.shape))).astype(np.int)
    return img[:,idx[0],idx[1]]
    
def align_nearest_neighbor_scaled_image(img, r):
    w, h = img.size
    return img.resize((int(w // r), int(h // r)), Image.NEAREST).resize((w, h), Image.NEAREST)

def pad_by_multiply_of(img, factor=64, add=0):
    img = np.asarray(img)
    h, w, c = img.shape
    nw = factor * math.ceil(w / factor)
    nh = factor * math.ceil(h / factor)
    ph = nh - h
    pw = nw - w
    img = np.pad(img, [
        (ph // 2 + add, (ph - ph // 2) + add),
        (pw // 2 + add, (pw - pw // 2) + add),
        (0, 0)
    ], mode='reflect')
    return Image.fromarray(img).convert('RGBA')

def transparent_background(img):
    background_color = img.getpixel((0, 0))
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if img.getpixel((i, j)) == background_color:
                img.putpixel((i, j), (0, 0, 0, 0))
    return img

# https://stackoverflow.com/a/24527424
def chunks(iterable, size):
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size - 1))

if __name__ == '__main__':
    for r in (1.5, 2, 3):
        print(r)
        h, w, c = 6, 12, 4
        x = np.arange(h * w * c).reshape((c, h, w))
        y = upsample_nearest_neighbor(x, r)
        assert y.shape == (c, h * r, w *r)
        for (i, j, k) in np.ndindex(y.shape):
            s = y[i, j, k]
            t = x[i, int(j // r), int(k // r)]
            assert s == t, '[{}, {}, {}], {} != {}'.format(i, j, k, s, t)
        z = downsample_nearest_neighbor(x, r)
        for (i, j, k) in np.ndindex(z.shape):
            s = z[i, j, k]
            t = x[i, int(j * r), int(k * r)]
            assert s == t, '[{}, {}, {}], {} != {}'.format(i, j, k, s, t)
        print(x, y, z)
