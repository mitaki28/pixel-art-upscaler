#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
from chainercv.transforms import resize

def out_image_base(updater, xp, n, seed, dst, converter):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)

        n_pattern = 3
        n_images = n * n_pattern

        rows = n
        cols = n_pattern
        
        w_in = 128
        w_out = 128
        in_ch = 4
        out_ch = 4
        
        ret = []
        
        for it in range(n):
            batch = updater.get_iterator('test').next()
            batchsize = len(batch)

            x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
            t_out = xp.zeros((batchsize, out_ch, w_out, w_out)).astype("f")

            for i in range(batchsize):
                x_in[i,:] = xp.asarray(batch[i][0])
                t_out[i,:] = xp.asarray(batch[i][1])

            x_in = Variable(x_in)
            x_out = converter(x_in)
            
            ret.append(chainer.cuda.to_cpu(x_in.data)[0,:])
            ret.append(chainer.cuda.to_cpu(t_out)[0,:])
            ret.append(chainer.cuda.to_cpu(x_out.data)[0,:])
        
        def save_image(x, name, mode=None):
            _, C, H, W = x.shape
            x = x.reshape((rows, cols, C, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            if C==1:
                x = x.reshape((rows*H, cols*W))
            else:
                x = x.reshape((rows*H, cols*W, C))

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image_{}_{:0>8}.png'.format(name, trainer.updater.iteration)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x, mode=mode).convert('RGBA').save(preview_path)
        
        ret = [resize(x, (w_out, w_out), Image.NEAREST) for x in ret]
        x = np.asarray(np.clip(np.asarray(ret) * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "gen")
        
    return make_image

def out_image(updater, enc, dec, n, seed, dst):
    def converter(x_in):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            return dec(enc(x_in))
    return out_image_base(updater, enc.xp, n, seed, dst, converter)

def convert_image_base(img, xp, converter):
        batchsize = 1
        w_in = 128
        w_out = 64

        img = (xp.asarray(img.convert('RGBA').resize((w_in, w_in), Image.NEAREST)).astype("f").transpose((2, 0, 1)) - 127.5) / 127.5

        x_in = Variable(img.reshape(1, *img.shape))
        x_out = converter(x_in)
        x = chainer.cuda.to_cpu(x_out.data)[0]

        C, H, W = x.shape
        x = x.transpose(1, 2, 0)
        if C==1:
            x = x.reshape((H, W))
        else:
            x = x.reshape((H, W, C))
        return Image.fromarray(
            np.asarray(xp.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        ).resize((w_out, w_out), Image.BOX)


def convert_image(img, enc, dec):
    def converter(x_in):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            return dec(enc(x_in))
    return convert_image_base(img, enc.xp, converter)
