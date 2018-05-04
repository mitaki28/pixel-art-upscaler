from PIL import Image
import numpy as np

import chainer
from pixcaler.util import chw_array_to_img, img_to_chw_array, align_nearest_neighbor_scaled_image, pad_by_multiply_of, chunks, upsample_nearest_neighbor, downsample_nearest_neighbor


class NullConversionEventHandler:
    def on_patch(self, patch, idx, n):
        pass

class Converter:
    def get_input_size(self):
        raise NotImplemented()        

    def __call__(self, img):
        raise NotImplemented()

class ChainerConverter(Converter):
    def __init__(self, gen, input_size):
        self.gen = gen
        self.input_size = input_size

    def get_input_size(self):
        return self.input_size

    def __call__(self, imgs):
        xp = self.gen.xp
        self.gen.fix_broken_batchnorm()
        
        x = xp.asarray([img_to_chw_array(img) for img in imgs])
        x_in = chainer.Variable(x)
        with chainer.using_config('train', False), chainer.using_config('enable_back_prop', False):
            x_out = self.gen(x_in)

        return [chw_array_to_img(x) for x in chainer.cuda.to_cpu(x_out.data)]

class PatchedExecuter:
    def __init__(self, converter, alignment_factor, batch_size, padding='half', handler=None):
        if padding == 'half':
            self.pad_size = converter.get_input_size() // 4
        elif padding == 'none':
            self.pad_size = 0
        else:
            assert False, 'Unknown padding type: {}'.format(padding)
        self.patch_size =  converter.get_input_size() - 2 * self.pad_size

        self.batch_size = batch_size
        self.converter = converter
        self.alignment_factor = alignment_factor
        self.handler = NullConversionEventHandler() if handler is None else handler

    def __call__(self, img):
        w_org, h_org = img.size
        img = pad_by_multiply_of(img, self.patch_size, self.pad_size)        
        n_i = (img.size[0] - self.pad_size * 2) // self.patch_size
        n_j = (img.size[1] - self.pad_size * 2) // self.patch_size
        converted_img = Image.new('RGBA', (
            img.size[0] - self.pad_size * 2,
            img.size[1] - self.pad_size * 2,
        ))
        def _patch_generator():            
            for i in range(n_i):
                for j in range(n_j):
                    x = i * self.patch_size
                    y = j * self.patch_size
                    yield (i, j, x, y), img.crop((
                        x,
                        y,
                        x + self.patch_size + 2 * self.pad_size,
                        y + self.patch_size + 2 * self.pad_size,
                    ))
        for chunk in chunks(_patch_generator(), self.batch_size):
            cords, batch = map(list, zip(*chunk))
            converted_batch = self.converter(batch)
            for (i, j, x, y), converted_patch in zip(cords, converted_batch):
                converted_img.paste(
                    converted_patch.crop((
                        self.pad_size, self.pad_size,
                        self.pad_size + self.patch_size, self.pad_size + self.patch_size,
                    )),
                    (x, y),
                )
                self.handler.on_patch(converted_patch, i * n_j + j, n_i * n_j)
        w_conv, h_conv = converted_img.size
        w_pad, h_pad = (w_conv - w_org, h_conv - h_org)
        return converted_img.crop((
            w_pad // 2,
            h_pad // 2,
            w_pad // 2 + w_org,
            h_pad // 2 + h_org
        ))        


class Upscaler:
    def __init__(self, converter, factor, batch_size=1, padding='half', handler=None):
        self.factor = factor
        self.executor = PatchedExecuter(
            converter,
            alignment_factor=factor,
            batch_size=batch_size,
            padding=padding,
            handler=handler,
        )

    def generate_comparable_image(img):
        return img.resize((int(img.size[0] * self.factor), int(img.size[1] * self.factor)), Image.BILINEAR)

    def __call__(self, img):
        img = img.resize((int(img.size[0] * self.factor), int(img.size[1] * self.factor)), Image.BILINEAR)
        return self.executor(img)

class Downscaler:
    def __init__(self, converter, factor=2, batch_size=1, handler=None):
        self.factor = factor
        self.executor = PatchedExecuter(
            converter,
            alignment_factor=1,
            batch_size=batch_size,
            handler=handler,
        )

    def generate_comparable_image(img):
        return chw_array_to_img(downsample_nearest_neighbor(img_to_chw_array(img), self.factor))

    def __call__(self, img):
        img = self.executor(img)
        return chw_array_to_img(downsample_nearest_neighbor(img_to_chw_array(img), self.factor))

class Refiner:
    def __init__(self, converter, batch_size=1, handler=None):
        self.executor = PatchedExecuter(
            converter,
            alignment_factor=1,
            batch_size=batch_size,
            handler=handler,
        )

    def generate_comparable_image(img):
        return img

    def __call__(self, img):
        return self.executor(img)

class MultiStageScaler:
    def __init__(self, scalers):
        self.scalers = scalers

    def generate_comparable_image(img):
        for scaler in self.scalers:
            img = scaler.generate_comparable_image(img)
        return img

    def __call__(self, img):
        for scaler in self.scalers:
            img = scaler(img)
        return img

class PrintLogger:
    def __init__(self, context=""):
        self.context = context
    def on_patch(self, patch, idx, n):
        print("{}: {}/{}".format(self.context, idx + 1, n), end='\r')