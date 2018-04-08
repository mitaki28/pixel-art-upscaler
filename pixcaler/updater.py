import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
from PIL import Image

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy

class Pix2PixUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('model')
        super().__init__(*args, **kwargs)

    def loss_func_adv_dis_fake_ls(self, y_fake):
        return 0.5 * F.mean(y_fake ** 2)

    def loss_func_adv_dis_real_ls(self, y_real):
        return 0.5 * F.mean((y_real - 1.0) ** 2)

    def loss_func_adv_gen_ls(self, y_fake):
        return 0.5 * F.mean((y_fake - 1.0) ** 2)

    def loss_func_adv_dis_fake(self, y_fake):
        return F.mean(F.softplus(y_fake))

    def loss_func_adv_dis_real(self, y_real):
        return F.mean(F.softplus(-y_real))

    def loss_func_adv_gen(self, y_fake):
        return F.mean(F.softplus(-y_fake))

    def loss_func_rec_gen(self, x_in, x_out):
        return F.mean_absolute_error(x_out, x_in)

    def loss_gen(self, enc, x_out, x_in, y_fake, lam1=10, lam2=1):
        loss_rec = lam1*self.loss_func_rec_gen(x_in, x_out)
        loss_adv = lam2*self.loss_func_adv_gen_ls(y_fake)
        loss = loss_rec + loss_adv
        chainer.report({'loss_rec': loss_rec}, enc)
        chainer.report({'loss_adv': loss_adv}, enc)
        chainer.report({'loss': loss}, enc)
        return loss

    def loss_dis(self, dis, y_real, y_fake):
        L1 = self.loss_func_adv_dis_real_ls(y_real)
        L2 = self.loss_func_adv_dis_fake_ls(y_fake)
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def update_core(self):        
        optimizer = self.get_optimizer('main')
        
        gen, dis = self.model.gen, self.model.dis
        xp = gen.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        x_in = Variable(xp.asarray([b[0] for b in batch]).astype('f'))
        t_out = Variable(xp.asarray([b[1] for b in batch]).astype('f'))
        

        x_out = gen(x_in)

        y_fake = dis(x_in, x_out)
        y_real = dis(x_in, t_out)

        gen.cleargrads()
        self.loss_gen(gen, x_out, t_out, y_fake).backward()
        x_in.unchain_backward()
        x_out.unchain_backward()

        dis.cleargrads()        
        self.loss_dis(dis, y_real, y_fake).backward()

        optimizer.update()

    
    def _debug(self, x, name):
        Image.fromarray(np.asarray(np.clip(np.asarray(x.data) * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8).reshape(x.data.shape[1:]).transpose((1, 2, 0))).save('result/preview/debug-{}.png'.format(name))

        

