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


def random_crop_by_2(img, C_label, pH, pW, fH, fW):
    y = np.random.randint(pH)
    x = np.random.randint(pW)
    label, data = img[:C_label], img[C_label:]
    label = label[:,y:y+fH,x:x+fW]
    y_data = (y // 2) * 2
    x_data = (x // 2) * 2
    data = data[:,y_data:y_data+fH,x_data:x_data+fW]
    return  np.concatenate([label, data], axis=0)

def convert_image(img):
    return np.asarray(img.convert('RGBA')).astype("f").transpose((2, 0, 1)) / 127.5 - 1.0

def argument_image(img, C_label, char_size, fine_size, is_crop_random=True, is_flip_random=True):
    cW, cH = char_size
    fW, fH = fine_size
    pW, pH = ((fW - cW), (fH - cH))
    if is_crop_random:
        assert pW >= 0 and pW % 2 == 0 and pH >= 0 and pH % 2 == 0
        img = resize_contain(img, (fH + pH, fW + pW), img[:,0,0])
        img = random_crop_by_2(img, C_label, pH, pW, fH, fW)
    else:
        img = resize_contain(img, (fH, fW), img[:,0,0])
    if is_flip_random:
        img = random_flip(img, x_random=True)
    return img


# TODO padding, resize は全部 Dataset 側でやるようにしたい
class PairDownscaleDataset(dataset_mixin.DatasetMixin):

    def __init__(self, dataDir, labelDir, charSize=(48, 48), fineSize=(64, 64)):
        self.charSize = charSize
        self.fineSize = fineSize
        self.dataDir = Path(dataDir)
        self.labelDir = Path(labelDir)
        data_names = set([path.name for path in self.dataDir.glob("*.png")])
        label_names = set([path.name for path in self.labelDir.glob("*.png")])
        self.filenames = list(data_names & label_names)
        print(len(self.filenames), 'loaded')
        print(len(label_names - data_names), 'ignored from label')
        print(len(data_names - label_names), 'ignored from data')
    
    def __len__(self):
        return len(self.filenames)

    # return (label, img)
    def get_example(self, i):
        filename = self.filenames[i]
        with Image.open(self.dataDir/filename) as f:
            img = convert_image(f)
        with Image.open(self.labelDir/filename) as f:
            label = convert_image(f)
        C, H, W = label.shape
        py, px = random.choice([(0, 0), (1, 0), (0, 1)])
        label[:,1::2,1::2] = label[:,py::2,px::2]
        C_label = label.shape[0]
        t = np.concatenate([label, img], axis=0)
        t = argument_image(t, C_label, self.charSize, self.fineSize)
        
        label, img = t[:C_label], t[C_label:]
        t = np.concatenate([label, img], axis=0)

        t = resize(t, (64, 64), Image.NEAREST)                
        return t[:C_label], t[C_label:]

def downscale_random_nearest_neighbor(img):
    c, h, w = img.shape
    img = img.reshape((c, h // 2, 2, w // 2, 2)).transpose((1, 3, 2, 4, 0)).reshape((h // 2, w // 2, 4, c))
    hw_idx = np.indices((h // 2, w // 2))
    c_idx = np.random.randint(0, 4, img.shape[:2])
    return img[hw_idx[0], hw_idx[1], c_idx].transpose((2, 0, 1))
    
class AutoUpscaleDataset(dataset_mixin.DatasetMixin):
    def __init__(self, labelDir, random_nn=True):
        self.labelDir = Path(labelDir)
        self.filepaths = list(self.labelDir.glob("*.png"))
        self.random_nn = random_nn
        print("{} images loaded".format(len(self.filepaths)))
    
    def __len__(self):
        return len(self.filepaths)

    # return (label, img)
    def get_example(self, i):
        with Image.open(str(self.filepaths[i])) as f:
            img = convert_image(f)

        # random background color
        # bgMask = ((-img + 1.0) / 2.0)[3,:,:]
        # bgMaskR = bgMask * (random() * 2.0 - 1.0)
        # bgMaskG = bgMask * (random() * 2.0 - 1.0)
        # bgMaskB = bgMask * (random() * 2.0 - 1.0)
        # img[0,:,:] += bgMaskR
        # img[1,:,:] += bgMaskG
        # img[2,:,:] += bgMaskB

        img = random_crop(img, (64, 64))
        img = random_flip(img, x_random=True)

        if self.random_nn:
            label = resize(downscale_random_nearest_neighbor(img), (64, 64), Image.NEAREST)
        else:
            label = resize(resize(img, (32, 32), Image.NEAREST), (64, 64), Image.NEAREST)
        img = resize(img, (64, 64), Image.NEAREST)
        return label, img

class AutoUpscaleDatasetReverse(AutoUpscaleDataset):

    def __init__(self, labelDir):
        self.labelDir = Path(labelDir)
        self.filepaths = list(self.labelDir.glob("*.png"))
        print("{} images loaded".format(len(self.filepaths)))

    def get_example(self, i):
        with Image.open(str(self.filepaths[i])) as f:
            label = convert_image(f)
        img = label.copy()
        C, H, W = label.shape
        # py, px = random.choice([(0, 0), (1, 0), (0, 1)])
        # label[:,1::2,1::2] = label[:,py::2,px::2]

        C_label = label.shape[0]
        t = np.concatenate([label, img], axis=0)


        # random background color
        # bgMask = ((-label + 1.0) / 2.0)[3,:,:]
        # bgMaskR = bgMask * (random() * 2.0 - 1.0)
        # bgMaskG = bgMask * (random() * 2.0 - 1.0)
        # bgMaskB = bgMask * (random() * 2.0 - 1.0)
        # label[0,:,:] += bgMaskR
        # label[1,:,:] += bgMaskG
        # label[2,:,:] += bgMaskB

        t = center_crop(t, (64, 64))
        t = random_flip(t, x_random=True)
        t = resize(t, (64, 64), Image.NEAREST)
        return t[:C_label], t[C_label:]
