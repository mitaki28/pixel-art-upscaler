from PIL import Image
import numpy as np

import chainer
from pixcaler.util import chw_array_to_img, img_to_chw_array, align_2x_nearest_neighbor_scaled_image, pad_by_multiply_of


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

    def __call__(self, img):
        xp = self.gen.xp
        
        x = xp.asarray(img_to_chw_array(img))
        x_in = chainer.Variable(x.reshape(1, *x.shape))
        with chainer.using_config('train', False), chainer.using_config('enable_back_prop', False):
            x_out = self.gen(x_in)

        return chw_array_to_img(chainer.cuda.to_cpu(x_out.data)[0])

class PatchedExecuter:
    def __init__(self, converter, is_alignment_required, handler=None):
        self.patch_size = converter.get_input_size() // 2
        self.converter = converter
        self.is_alignment_required = is_alignment_required
        self.handler = NullConversionEventHandler() if handler is None else handler

    def __call__(self, img):
        ps = self.patch_size
        w_org, h_org = img.size
        img = pad_by_multiply_of(img, ps, ps // 2)
        if self.is_alignment_required:
            img = align_2x_nearest_neighbor_scaled_image(img)
        
        n_i = (img.size[0] - ps // 2 * 2) // ps
        n_j = (img.size[1] - ps // 2 * 2) // ps
        converted_img = Image.new('RGBA', (
            img.size[0] - ps // 2 * 2,
            img.size[1] - ps // 2 * 2,
        ))        
        for i in range(n_i):
            for j in range(n_j):
                x = i * ps
                y = j * ps
                patch = img.crop((
                    x,
                    y,
                    x + 2 * ps,
                    y + 2 * ps,
                ))
                converted_patch = self.converter(patch)
                converted_img.paste(
                    converted_patch.crop((
                        ps // 2, ps // 2,
                        ps // 2 + ps, ps // 2 + ps,
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
    def __init__(self, converter, handler=None):
        self.executor = PatchedExecuter(
            converter,
            is_alignment_required=True,
            handler=handler,
        )

    def generate_comparable_image(img):
        return img.resize((img.size[0] * 2, img.size[1] * 2), Image.NEAREST)        

    def __call__(self, img):
        img = img.resize((img.size[0] * 2, img.size[1] * 2), Image.NEAREST)
        return self.executor(img)

class Downscaler:
    def __init__(self, converter, handler=None):
        self.executor = PatchedExecuter(
            converter,
            is_alignment_required=False,
            handler=handler,
        )

    def generate_comparable_image(img):
        return img.resize((img.size[0] // 2, img.size[1] // 2), Image.NEAREST)

    def __call__(self, img):
        img = self.executor(img)
        return img.resize((img.size[0] // 2, img.size[1] // 2), Image.NEAREST)