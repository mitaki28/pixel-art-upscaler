#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
from PIL import Image

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy

class FacadeUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.enc, self.dec, self.dis = kwargs.pop('models')
        super(FacadeUpdater, self).__init__(*args, **kwargs)

    def loss_func_adv_dis_fake_ls(self, y_fake):
        return 0.5 * F.mean(y_fake ** 2)

    def loss_func_adv_dis_real_ls(self, y_real):
        return 0.5 * F.mean((y_real - 1.0) ** 2)

    def loss_func_adv_gen_ls(self, y_fake):
        return 0.5 * F.mean((y_fake - 1.0) ** 2)

    def loss_func_adv_dis_fake(self, y_fake):
        batchsize,_,w,h = y_fake.data.shape
        return F.sum(F.softplus(y_fake)) / batchsize / w / h

    def loss_func_adv_dis_real(self, y_real):
        batchsize,_,w,h = y_real.data.shape
        return F.sum(F.softplus(-y_real)) / batchsize / w / h

    def loss_func_adv_gen(self, y_fake):
        batchsize,_,w,h = y_fake.data.shape
        return F.sum(F.softplus(-y_fake)) / batchsize / w / h

    def loss_func_rec_gen(self, x_in, x_out):
        return F.mean_absolute_error(x_out, x_in)

    def loss_gen(self, enc, x_out, x_in, y_fake, lam1=100, lam2=1):
        loss_rec = lam1*self.loss_func_rec_gen(x_in, x_out)
        loss_adv = lam2*self.loss_func_adv_gen(y_fake)
        loss = loss_rec + loss_adv
        chainer.report({'loss_rec': loss_rec}, enc)
        chainer.report({'loss_adv': loss_adv}, enc)
        chainer.report({'loss': loss}, enc)
        return loss

    def loss_dis(self, dis, y_real, y_fake):
        L1 = self.loss_func_adv_dis_real(y_real)
        L2 = self.loss_func_adv_dis_fake(y_fake)
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def update_core(self):        
        enc_optimizer = self.get_optimizer('enc')
        dec_optimizer = self.get_optimizer('dec')
        dis_optimizer = self.get_optimizer('dis')
        
        enc, dec, dis = self.enc, self.dec, self.dis
        xp = enc.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        x_in = Variable(xp.asarray([b[0] for b in batch]).astype('f'))
        t_out = Variable(xp.asarray([b[1] for b in batch]).astype('f'))
        
        z = enc(x_in)
        x_out = dec(z)

        y_fake = dis(x_in, x_out)
        y_real = dis(x_in, t_out)


        enc_optimizer.update(self.loss_gen, enc, x_out, t_out, y_fake)
        for z_ in z:
            z_.unchain_backward()
        dec_optimizer.update(self.loss_gen, dec, x_out, t_out, y_fake)
        x_in.unchain_backward()
        x_out.unchain_backward()
        dis_optimizer.update(self.loss_dis, dis, y_real, y_fake)

    
    def _debug(self, x, name):
        Image.fromarray(np.asarray(np.clip(np.asarray(x.data) * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8).reshape(x.data.shape[1:]).transpose((1, 2, 0))).save('result/preview/debug-{}.png'.format(name))

        

