import numpy as np

import random
from PIL import Image
from pathlib import Path
from chainercv.transforms import center_crop
from chainercv.transforms import random_crop
from chainercv.transforms import random_flip
from chainercv.transforms import resize_contain
from chainercv.transforms import resize
from chainercv.utils import read_image

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
        np.asarray(xp.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    )

def downscale_random_nearest_neighbor(img):
    c, h, w = img.shape
    img = img.reshape((c, h // 2, 2, w // 2, 2)).transpose((1, 3, 2, 4, 0)).reshape((h // 2, w // 2, 4, c))
    hw_idx = np.indices((h // 2, w // 2))
    c_idx = np.random.randint(0, 4, img.shape[:2])
    return img[hw_idx[0], hw_idx[1], c_idx].transpose((2, 0, 1))