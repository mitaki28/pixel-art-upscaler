import numpy as np
from pathlib import Path
import random
import math

from PIL import Image
from PIL import ImageFont, ImageDraw

from chainer.dataset import dataset_mixin
from chainercv.transforms import center_crop
from chainercv.transforms import random_crop
from chainercv.transforms import random_flip
from chainercv.transforms import resize_contain
from chainercv.transforms import resize
from chainercv.utils import read_image

from pixcaler.util import img_to_chw_array, downscale_random_nearest_neighbor

def random_crop_by_2(img, c_source, pH, pW, fH, fW):
    y = np.random.randint(pH)
    x = np.random.randint(pW)
    source, target = img[:c_source], img[c_source:]
    source = source[:,y:y+fH,x:x+fW]
    y_target = (y // 2) * 2
    x_target = (x // 2) * 2
    target = target[:,y_target:y_target+fH,x_target:x_target+fW]
    return np.concatenate([source, target], axis=0)

# TODO padding, resize は全部 Dataset 側でやるようにしたい
class PairDownscaleDataset(dataset_mixin.DatasetMixin):

    def __init__(self, target_dir, source_dir, char_size=(48, 48), fine_size=(64, 64)):
        self.char_size = char_size
        self.fine_size = fine_size
        self.target_dir = Path(target_dir)
        self.source_dir = Path(source_dir)
        target_names = set([path.name for path in self.target_dir.glob("*.png")])
        source_names = set([path.name for path in self.source_dir.glob("*.png")])
        self.filenames = list(target_names & source_names)
        print(len(self.filenames), 'loaded')
        print(len(source_names - target_names), 'ignored from source')
        print(len(target_names - source_names), 'ignored from target')
    
    def __len__(self):
        return len(self.filenames)

    def argument_image(self, img, c_source, is_crop_random=True, is_flip_random=True):
        cW, cH = self.char_size
        fW, fH = self.fine_size
        pW, pH = ((fW - cW), (fH - cH))
        if is_crop_random:
            assert pW >= 0 and pW % 2 == 0 and pH >= 0 and pH % 2 == 0
            img = resize_contain(img, (fH + pH, fW + pW), img[:,0,0])
            img = random_crop_by_2(img, c_source, pH, pW, fH, fW)
        else:
            img = resize_contain(img, (fH, fW), img[:,0,0])
        if is_flip_random:
            img = random_flip(img, x_random=True)
        return img

    # return (source, img)
    def get_example(self, i):
        filename = self.filenames[i]
        with Image.open(self.source_dir/filename) as f:
            source = img_to_chw_array(f)
        with Image.open(self.target_dir/filename) as f:
            target = img_to_chw_array(f)
        C, H, W = source.shape
        py, px = random.choice([(0, 0), (1, 0), (0, 1)])
        source[:,1::2,1::2] = source[:,py::2,px::2]
        
        c_source = source.shape[0]
        t = np.concatenate([source, target], axis=0)
        t = self.argument_image(t, c_source, self.char_size, self.fine_size)

        return t[:c_source], t[c_source:]
    
class AutoUpscaleDataset(dataset_mixin.DatasetMixin):
    def __init__(self, target_dir, random_nn=False, fine_size=64):
        self.target_dir = Path(target_dir)
        self.filepaths = list(self.target_dir.glob("*.png"))
        self.random_nn = random_nn
        self.fine_size = fine_size
        print("{} images loaded".format(len(self.filepaths)))
    
    def __len__(self):
        return len(self.filepaths)

    # return (source, target)
    def get_example(self, i):
        with Image.open(str(self.filepaths[i])) as f:
            target = img_to_chw_array(f)

        target = random_crop(target, (self.fine_size, self.fine_size))
        target = random_flip(target, x_random=True)
        if self.random_nn:
            source = resize(
                downscale_random_nearest_neighbor(target.copy()),
                (self.fine_size, self.fine_size), Image.NEAREST,
            )
        else:
            source = resize(
                resize(
                    target,
                    (self.fine_size // 2, self.fine_size // 2), Image.NEAREST,
                ),
                (self.fine_size, self.fine_size), Image.NEAREST,
            )
        return source, target

class CompositeAutoUpscaleDataset(dataset_mixin.DatasetMixin):
    def __init__(self, data_dir, fine_size=64):
        import pixcaler.charset
        self.data_dir = Path(data_dir)
        self.chartips = list((self.data_dir/'chartip').glob("*.png"))
        self.tiles = list((self.data_dir/'tile').glob("*.png"))
        self.objs = list((self.data_dir/'obj').glob("*.png"))
        self.fonts = list((self.data_dir/'font').glob("*.ttf"))
        self.charset = pixcaler.charset.ALL

        self.fine_size = fine_size
        print("{} chartips loaded".format(len(self.chartips)))
        print("{} tiles loaded".format(len(self.tiles)))
        print("{} objs loaded".format(len(self.objs)))
        print("{} fonts loaded".format(len(self.fonts)))
    
    def __len__(self):
        return 10000

    # return (source, target)
    def get_example(self, i):
        r = random.random()
        if r < 0.5:
            with Image.open(str(self.chartips[np.random.randint(len(self.chartips))])) as img:
                front = img_to_chw_array(img)
        elif r < 0.75:
            with Image.open(str(self.objs[np.random.randint(len(self.objs))])) as img:
                front = img_to_chw_array(img)
        else:
            S = 1
            w, h = self.fine_size * 2 * S, self.fine_size * 2 * S
            front = Image.new('RGBA', (w, h))
            draw = ImageDraw.Draw(front)
            fontsize = np.random.randint(32, 64) * S
            font = ImageFont.truetype(str(self.fonts[np.random.randint(len(self.fonts))]), fontsize * 3 // 4)
            txt = ''
            for i in range(math.ceil(fontsize / h) * 4):
                txt += ''.join(random.choices(self.charset, k=math.ceil(fontsize / w) * 4))
                txt += '\n'
            r, g, b = [np.random.randint(256) for i in range(3)]
            draw.text((0, 0), txt, font=font, fill=(r, g, b, 255))
            front = img_to_chw_array(front)
        front = random_crop(front, (self.fine_size, self.fine_size))
        front = random_flip(front, x_random=True)
        
        r = random.random()
        if r < 0.9:
            with Image.open(str(self.tiles[np.random.randint(len(self.tiles))])) as img:
                back = img_to_chw_array(img)
        else:
            r, g, b = [np.random.randint(256) for i in range(3)]
            back = Image.new('RGBA', (self.fine_size, self.fine_size), (r, g, b))
            back = img_to_chw_array(back)
        back = random_crop(back, (self.fine_size, self.fine_size))
        back = random_flip(back, x_random=True)
        m = np.tile((front[3,:,:] == 1).reshape((1, self.fine_size, self.fine_size)), (4, 1, 1)).reshape(4 * self.fine_size ** 2)
        back = back.reshape(4 * self.fine_size ** 2)
        back[m] = front.reshape(4 * self.fine_size ** 2)[m]
        target = back.reshape((4, self.fine_size, self.fine_size))
        source = resize(
            resize(
                target,
                (self.fine_size // 2, self.fine_size // 2), Image.NEAREST,
            ),
            (self.fine_size, self.fine_size), Image.NEAREST,
        )
        return source, target

class AutoUpscaleDatasetReverse(AutoUpscaleDataset):
    def get_example(self, i):
        source, target = super().get_example(i)
        return target, source

class Single32Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, target_dir, random_nn=False, fine_size=64):
        self.target_dir = Path(target_dir)
        self.filepaths = list(self.target_dir.glob("*.png"))
        self.fine_size = fine_size
        print("{} images loaded".format(len(self.filepaths)))
    
    def __len__(self):
        return len(self.filepaths)

    # return (source, target)
    def get_example(self, i):
        with Image.open(str(self.filepaths[i])) as f:
            target = img_to_chw_array(f)

        target = random_crop(target, (self.fine_size, self.fine_size))
        target = random_flip(target, x_random=True)
        # randomize and alignment
        source = resize(
            downscale_random_nearest_neighbor(target.copy()),
            (self.fine_size, self.fine_size), Image.NEAREST,
        )
        # alignment
        target = resize(
            resize(
                target,
                (self.fine_size // 2, self.fine_size // 2), Image.NEAREST,
            ),
            (self.fine_size, self.fine_size), Image.NEAREST,
        )
        return source, target
