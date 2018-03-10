#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable

def out_image_base(updater, xp, rows, cols, seed, dst, converter):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        
        w_in = 64
        w_out = 64
        in_ch = 4
        out_ch = 4
        
        in_all = np.zeros((n_images, in_ch, w_in, w_in)).astype("f")
        gt_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")
        gen_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")
        
        for it in range(n_images):
            batch = updater.get_iterator('test').next()
            batchsize = len(batch)

            x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
            t_out = xp.zeros((batchsize, out_ch, w_out, w_out)).astype("f")

            for i in range(batchsize):
                x_in[i,:] = xp.asarray(batch[i][0])
                t_out[i,:] = xp.asarray(batch[i][1])

            x_in = Variable(x_in)
            x_out = converter(x_in)
            
            in_all[it,:] = x_in.data[0,:]
            gt_all[it,:] = t_out[0,:]
            gen_all[it,:] = x_out.data[0,:]
        
        
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
        
        x = np.asarray(np.clip(gen_all * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "gen")
        
        x = np.asarray(np.clip(in_all * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "in")
        
        x = np.asarray(np.clip(gt_all * 127.5+127.5, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "gt")
        
    return make_image

def out_image(updater, enc, dec, rows, cols, seed, dst):
    def converter(x_in):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            return dec(enc(x_in))
    return out_image_base(updater, enc.xp, rows, cols, seed, dst, converter)

def out_image2(updater, enc0, dec0, enc, dec, rows, cols, seed, dst):
    def converter(x_in):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            return dec(enc(dec0(enc0(x_in))))
    return out_image_base(updater, enc.xp, rows, cols, seed, dst, converter)

def convert_image_base(imgs, xp, converter):
        batchsize = 4
        w_in = 64
        w_out = 64
        #w_out, h_out = imgs[0].size
        #assert w_out == h_out
        imgs = np.asarray([
            (np.asarray(img.convert('RGBA').resize((w_in, w_in), Image.NEAREST)).astype("f").transpose((2, 0, 1)) - 127.5) / 127.5
            for img
            in imgs
        ])

        xs = []
        for i in range((len(imgs) - 1) // batchsize + 1):
            imgs_chunk = imgs[i*batchsize:(i+1)*batchsize]
            x_in = Variable(imgs_chunk)
            x_out = converter(x_in)
            xs.extend(x_out.data)
            print((i + 1) * batchsize, 'images done')

        ret = []
        for x in xs:
            C, H, W = x.shape
            x = x.transpose(1, 2, 0)
            if C==1:
                x = x.reshape((H, W))
            else:
                x = x.reshape((H, W, C))
            ret.append(Image.fromarray(
                np.asarray(xp.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
            ).resize((w_out, w_out), Image.NEAREST))
        return ret


def convert_image(imgs, enc, dec):
    def converter(x_in):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            return dec(enc(x_in))
    return convert_image_base(imgs, enc.xp, converter)


def convert_image2(imgs, enc0, dec0, enc, dec):
    def converter(x_in):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            return dec(enc(dec0(enc0(x_in))))
    return convert_image_base(imgs, enc.xp, converter)
