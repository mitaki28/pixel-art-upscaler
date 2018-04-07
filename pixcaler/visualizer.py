import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
from chainercv.transforms import resize, resize_contain
from pixcaler.util import chw_array_to_img

def out_image(gen, n, dst):
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
            assert batchsize == 1

            x_in = Variable(xp.asarray([b[0] for b in batch]).astype('f'))
            t_out = Variable(xp.asarray([b[1] for b in batch]).astype('f'))
            x_out, _ = gen(x_in)
    
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


