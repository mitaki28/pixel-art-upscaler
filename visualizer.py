#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
from chainercv.transforms import resize, resize_contain

def out_image_base(gen, n, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        xp = gen.xp
        n_pattern = 3
        n_images = n * n_pattern

        rows = n
        cols = n_pattern
        
        ret = []
        
        for it in range(n):
            batch = trainer.updater.get_iterator('test').next()
            batchsize = len(batch)

            x_in = Variable(xp.asarray([b[0] for b in batch]).astype('f'))
            t_out = Variable(xp.asarray([b[1] for b in batch]).astype('f'))
            x_out, _ = gen(x_in)
    
            x_in = chainer.cuda.to_cpu(x_in.data)[0,:]
            t_out = chainer.cuda.to_cpu(t_out.data)[0,:]
            x_out = chainer.cuda.to_cpu(x_out.data)[0,:]
            ret.append(x_in)
            ret.append(t_out)
            ret.append(x_out)
        
        x = np.asarray(np.clip(np.asarray(ret) * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)

        _, C, H, W = x.shape
        x = x.reshape((rows, cols, C, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        if C==1:
            x = x.reshape((rows*H, cols*W))
        else:
            x = x.reshape((rows*H, cols*W, C))
        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image_{:0>8}.png'.format(trainer.updater.iteration)
        current_path = preview_dir + '/image_current.png'
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        img = Image.fromarray(x, mode=mode).convert('RGBA')
        img.save(preview_path)
        img.save(current_path)
    return make_image

def convert_image(img, gen):
        xp = gen.xp
        batchsize = 1
        img = (xp.asarray(img).astype("f").transpose((2, 0, 1)) - 127.5) / 127.5

        x_in = Variable(img.reshape(1, *img.shape))
        x_out, _ = gen(x_in)
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

