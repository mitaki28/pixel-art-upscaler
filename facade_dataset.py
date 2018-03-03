import numpy as np

from PIL import Image
from chainer.dataset import dataset_mixin
from pathlib import Path

class FacadeDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir, labelDir):
        self.dataDir = Path(dataDir)
        self.labelDir = Path(labelDir)
        self.dataset = []
        for filepath in self.dataDir.glob("*.png"):
            with Image.open(self.dataDir/filepath.name) as f:
                img = np.asarray(f.resize((128, 128), Image.NEAREST)).astype("f").transpose(2,0,1)/128.0-1.0                
            with Image.open(self.labelDir/filepath.name) as f:
                label = np.asarray(f.resize((128, 128), Image.NEAREST)).astype("f").transpose(2,0,1)/128.0-1.0
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
            with Image.open(self.labelDir/filepath.name) as f:
                img = np.asarray(f.resize((32, 32), Image.NEAREST).resize((128, 128), Image.NEAREST)).astype("f").transpose(2,0,1)/128.0-1.0
                label = np.asarray(f.resize((128, 128), Image.NEAREST)).astype("f").transpose(2,0,1)/128.0-1.0
            self.dataset.append((img,label))
        print("load dataset done")
    
    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i):
        return self.dataset[i][1], self.dataset[i][0]
    