import numpy as np

from random import random
from PIL import Image
from chainer.dataset import dataset_mixin
from pathlib import Path
from chainercv.transforms import random_crop
from chainercv.transforms import random_flip
from chainercv.transforms import resize_contain
from chainercv.utils import read_image

def convert_image(img):
    return np.asarray(img.convert('RGBA')).astype("f").transpose((2, 0, 1)) / 127.5 - 1.0

def argument_image(x, char_size, fine_size, is_crop_random=True, is_flip_random=True):
    cW, cH = char_size
    fW, fH = fine_size
    pW, pH = ((cW - fW), (cH - fH))
    if is_crop_random:
        assert pW >= 0 and pW % 2 == 0 and pH >= 0 and pH % 2 == 0
        x = resize_contain(x, (fW + pW, fH + pH), x[:,0,0])
        x = random_crop(x, (fW, fH))
    else:
        x = resize_contain(x, (fW, fH), x[:,0,0])
    if is_flip_random:
        x = random_flip(x, x_random=True)
    return x


class PairDataset(dataset_mixin.DatasetMixin):

    def __init__(self, dataDir, labelDir, charSize=(48, 64), fineSize=(64, 64)):
        self.charSize = charSize
        self.fineSize = fineSize
        self.dataDir = Path(dataDir)
        self.labelDir = Path(labelDir)
        data_names = set([path.name for path in self.dataDir.glob("*.png")])
        label_names = set([path.name for path in self.labelDir.glob("*.png")])
        self.filenames = list(data_names & label_names)
    
    def __len__(self):
        return len(self.filepaths)

    # return (label, img)
    def get_example(self, i):
        filename = self.filenames[i]
        with Image.open(self.dataDir/filename) as f:
            img = convert_image(f)
        with Image.open(self.labelDir/filename) as f:
            label = convert_image(f)
        C_label = label.shape[0]
        t = argument_image(np.concatenate(label, img), self.charSize, self.fineSize)
        t = resize(img, (128, 128), Image.NEAREST)
        return t[:C_label], t[C_label:]
    
class NNDownscaleDataset(dataset_mixin.DatasetMixin):
    def __init__(self, labelDir):
        self.labelDir = Path(labelDir)
        self.filepaths = list(self.labelDir.glob("*.png"))
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

        label = resize(resize(img, (32, 32), Image.NEAREST), (128, 128), Image.NEAREST)
        img = resize(img, (128, 128), Image.NEAREST)
        return label, img
