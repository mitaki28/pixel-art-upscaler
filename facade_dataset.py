import numpy as np

from random import random
from PIL import Image
from chainer.dataset import dataset_mixin
from pathlib import Path
from chainercv.transforms import random_crop
from chainercv.transforms import random_flip
from chainercv.transforms import resize
from chainercv.utils import read_image

class FacadeDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir, labelDir):
        self.dataDir = Path(dataDir)
        self.labelDir = Path(labelDir)
        self.dataset = []
        for filepath in self.dataDir.glob("*.png"):
            with Image.open(self.dataDir/filepath.name) as f:
                img = np.asarray(f.resize((64, 64), Image.NEAREST)).astype("f").transpose(2,0,1)/127.5-1.0                
            with Image.open(self.labelDir/filepath.name) as f:
                label = np.asarray(f.resize((64, 64), Image.NEAREST)).astype("f").transpose(2,0,1)/127.5-1.0
            self.dataset.append((img,label))
        print("load dataset done")
    
    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i):
        return self.dataset[i][1], self.dataset[i][0]
    
class HiResoDataset(dataset_mixin.DatasetMixin):
    def __init__(self, labelDir):
        self.labelDir = Path(labelDir)
        self.dataset = []
        for filepath in self.labelDir.glob("*.png"):
            with Image.open(str(filepath)) as f:
                img = np.asarray(f.convert('RGBA')).astype("f").transpose((2, 0, 1))
                self.dataset.append(img / 127.5 - 1.0)
        print("{} images loaded".format(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i):
        img = self.dataset[i]

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
    