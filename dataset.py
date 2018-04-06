import numpy as np

import random
from PIL import Image, ImageFilter
from chainer.dataset import dataset_mixin
from pathlib import Path
from chainercv.transforms import center_crop
from chainercv.transforms import random_crop
from chainercv.transforms import random_flip
from chainercv.transforms import resize_contain
from chainercv.transforms import resize
from chainercv.utils import read_image

from util import img_to_chw_array, downscale_random_nearest_neighbor

def random_crop_by_2(img, c_label, pH, pW, fH, fW):
    y = np.random.randint(pH)
    x = np.random.randint(pW)
    label, data = img[:c_label], img[c_label:]
    label = label[:,y:y+fH,x:x+fW]
    y_data = (y // 2) * 2
    x_data = (x // 2) * 2
    data = data[:,y_data:y_data+fH,x_data:x_data+fW]
    return np.concatenate([label, data], axis=0)

# TODO padding, resize は全部 Dataset 側でやるようにしたい
class PairDownscaleDataset(dataset_mixin.DatasetMixin):

    def __init__(self, data_dir, label_dir, char_size=(48, 48), fine_size=(64, 64)):
        self.char_size = char_size
        self.fine_size = fine_size
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        data_names = set([path.name for path in self.data_dir.glob("*.png")])
        label_names = set([path.name for path in self.label_dir.glob("*.png")])
        self.filenames = list(data_names & label_names)
        print(len(self.filenames), 'loaded')
        print(len(label_names - data_names), 'ignored from label')
        print(len(data_names - label_names), 'ignored from data')
    
    def __len__(self):
        return len(self.filenames)

    def argument_image(self, img, c_label, is_crop_random=True, is_flip_random=True):
        cW, cH = self.char_size
        fW, fH = self.fine_size
        pW, pH = ((fW - cW), (fH - cH))
        if is_crop_random:
            assert pW >= 0 and pW % 2 == 0 and pH >= 0 and pH % 2 == 0
            img = resize_contain(img, (fH + pH, fW + pW), img[:,0,0])
            img = random_crop_by_2(img, c_label, pH, pW, fH, fW)
        else:
            img = resize_contain(img, (fH, fW), img[:,0,0])
        if is_flip_random:
            img = random_flip(img, x_random=True)
        return img

    # return (label, img)
    def get_example(self, i):
        filename = self.filenames[i]
        with Image.open(self.data_dir/filename) as f:
            img = img_to_chw_array(f)
        with Image.open(self.label_dir/filename) as f:
            label = img_to_chw_array(f)
        C, H, W = label.shape
        py, px = random.choice([(0, 0), (1, 0), (0, 1)])
        label[:,1::2,1::2] = label[:,py::2,px::2]
        
        c_label = label.shape[0]
        t = np.concatenate([label, img], axis=0)
        t = self.argument_image(t, c_label, self.char_size, self.fine_size)

        t = resize(t, (64, 64), Image.NEAREST)                
        return t[:c_label], t[c_label:]
    
class AutoUpscaleDataset(dataset_mixin.DatasetMixin):
    def __init__(self, label_dir, random_nn=True):
        self.label_dir = Path(label_dir)
        self.filepaths = list(self.label_dir.glob("*.png"))
        self.random_nn = random_nn
        print("{} images loaded".format(len(self.filepaths)))
    
    def __len__(self):
        return len(self.filepaths)

    # return (label, img)
    def get_example(self, i):
        with Image.open(str(self.filepaths[i])) as f:
            img = img_to_chw_array(f)

        img = random_crop(img, (64, 64))
        img = random_flip(img, x_random=True)
        if self.random_nn:
            label = resize(downscale_random_nearest_neighbor(img), (64, 64), Image.NEAREST)
        else:
            label = resize(resize(img, (32, 32), Image.NEAREST), (64, 64), Image.NEAREST)
        img = resize(img, (64, 64), Image.NEAREST)
        return label, img

class AutoUpscaleDatasetReverse(AutoUpscaleDataset):

    def __init__(self, label_dir):
        self.label_dir = Path(label_dir)
        self.filepaths = list(self.label_dir.glob("*.png"))
        print("{} images loaded".format(len(self.filepaths)))

    def get_example(self, i):
        with Image.open(str(self.filepaths[i])) as f:
            label = img_to_chw_array(f)
        img = label.copy()
        C, H, W = label.shape
        # py, px = random.choice([(0, 0), (1, 0), (0, 1)])
        # label[:,1::2,1::2] = label[:,py::2,px::2]

        c_label = label.shape[0]
        t = np.concatenate([label, img], axis=0)

        t = center_crop(t, (64, 64))
        t = random_flip(t, x_random=True)
        t = resize(t, (64, 64), Image.NEAREST)
        return t[:c_label], t[c_label:]
