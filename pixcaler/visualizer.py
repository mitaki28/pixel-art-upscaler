import os

import numpy as np
from PIL import Image
from pathlib import Path

import chainer
import chainer.cuda
from chainer import Variable
from chainercv.transforms import resize, resize_contain
from pixcaler.util import chw_array_to_img

def out_image(test_iter, gen, n, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        xp = gen.xp
        n_pattern = 3
        n_images = n * n_pattern

        rows = n
        cols = n_pattern
        
        ret = []
        
        for it in range(n):
            batch = test_iter.next()

            x_in = Variable(xp.asarray([b[0] for b in batch]).astype('f'))
            t_out = Variable(xp.asarray([b[1] for b in batch]).astype('f'))
            x_out = gen(x_in)
    
            x_in = chainer.cuda.to_cpu(x_in.data)[0,:]
            t_out = chainer.cuda.to_cpu(t_out.data)[0,:]
            x_out = chainer.cuda.to_cpu(x_out.data)[0,:]
            ret.append(x_in)
            ret.append(t_out)
            ret.append(x_out)
        
        C, H, W = ret[0].shape
        x = np.asarray(ret).reshape((rows, cols, C, H, W)).transpose((2, 0, 3, 1, 4)).reshape((C, rows*H, cols*W))
        
        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image_{:0>8}.png'.format(trainer.updater.iteration)
        current_path = preview_dir + '/image_current.png'
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        img = chw_array_to_img(x)
        img.save(preview_path)
        img.save(current_path)
    return make_image

def full_out_image(scaler, src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)/'preview'
    latest_dir = dst_dir/'latest'
    @chainer.training.make_extension()
    def make_image(trainer):
        out_dir = dst_dir/'{:0>8}'.format(trainer.updater.iteration)
        out_dir.mkdir(exist_ok=True, parents=True)
        for path in sorted(src_dir.glob("*.png"), key=lambda p: p.name):
            with Image.open(str(path)) as img:
                scaler(img.convert('RGBA')).save(out_dir/path.name)
        try:
            latest_dir.unlink()
        except FileNotFoundError:
            pass
        latest_dir.symlink_to(out_dir, target_is_directory=True)
    return make_image


def out_image_cycle(gen_up, gen_down, n, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        xp = gen_up.xp
        n_pattern = 6
        n_images = n * n_pattern

        rows = n
        cols = n_pattern
        
        ret = []
        
        gen_up.fix_broken_batchnorm()
        gen_down.fix_broken_batchnorm()
        for it in range(n):
            batch_l = trainer.updater.get_iterator('testA').next()
            batch_s = trainer.updater.get_iterator('testB').next()
            x_l = Variable(xp.asarray([b[1] for b in batch_l]).astype('f'))
            x_s = Variable(xp.asarray([b[0] for b in batch_s]).astype('f'))
            with chainer.using_config('train', False), chainer.using_config('enable_back_prop', False):
                x_ls = gen_down(x_l)
                x_lsl = gen_up(x_ls)
                x_sl = gen_up(x_s)
                x_sls = gen_down(x_sl)
    
            x_l = chainer.cuda.to_cpu(x_l.data)[0,:]
            x_ls = chainer.cuda.to_cpu(x_ls.data)[0,:]
            x_lsl = chainer.cuda.to_cpu(x_lsl.data)[0,:]
            x_s = chainer.cuda.to_cpu(x_s.data)[0,:]
            x_sl = chainer.cuda.to_cpu(x_sl.data)[0,:]
            x_sls = chainer.cuda.to_cpu(x_sls.data)[0,:]
            ret.append(x_l)
            ret.append(x_ls)
            ret.append(x_lsl)
            ret.append(x_s)
            ret.append(x_sl)
            ret.append(x_sls)
        
        C, H, W = ret[0].shape
        x = np.asarray(ret).reshape((rows, cols, C, H, W)).transpose((2, 0, 3, 1, 4)).reshape((C, rows*H, cols*W))
        
        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image_{:0>8}.png'.format(trainer.updater.iteration)
        current_path = preview_dir + '/image_current.png'
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        img = chw_array_to_img(x)
        img.save(preview_path)
        img.save(current_path)
    return make_image
