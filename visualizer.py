#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
from chainercv.transforms import resize, resize_contain

def out_image_base(updater, xp, n, seed, dst, converter):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)

        n_pattern = 4
        n_images = n * n_pattern

        rows = n
        cols = n_pattern
        
        ret = []
        
        for it in range(n):
            batch = updater.get_iterator('test').next()
            batchsize = len(batch)

            x_in = Variable(xp.asarray([b[0] for b in batch]).astype('f'))
            t_out = Variable(xp.asarray([b[1] for b in batch]).astype('f'))
            x_out = converter(x_in)
    
            x_in = chainer.cuda.to_cpu(x_in.data)[0,:]
            t_out = chainer.cuda.to_cpu(t_out.data)[0,:]
            x_out = chainer.cuda.to_cpu(x_out.data)[0,:]
            ret.append(x_in)
            ret.append(t_out)
            ret.append(x_out)
            _, h_out, w_out = x_out.shape
            ret.append(resize(resize(x_out, (h_out // 2, w_out // 2), Image.BOX), (h_out, w_out), Image.NEAREST))
        
        def save_image(x, name, mode=None):
            _, C, H, W = x.shape
            x = x.reshape((rows, cols, C, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            if C==1:
                x = x.reshape((rows*H, cols*W))
            else:
                x = x.reshape((rows*H, cols*W, C))

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir + '/image_{}_{:0>8}.png'.format(name, trainer.updater.iteration)
            current_path = preview_dir + '/image_{}_cureent.png'.format(name)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            img = Image.fromarray(x, mode=mode).convert('RGBA')
            img.save(preview_path)
            img.save(current_path)
        max_h = max(r.shape[1] for r in ret)
        max_w = max(r.shape[2] for r in ret)
        ret = [resize_contain(x, (max_h, max_w), x[:,0,0]) for x in ret]
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
        w_in = 64
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
        )


def convert_image(img, enc, dec):
    def converter(x_in):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            return dec(enc(x_in))
    return convert_image_base(img, enc.xp, converter)
