import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
from PIL import Image

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy

def _debug(x, name):
    Image.fromarray(np.asarray(np.clip(np.asarray(x.data) * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8).reshape(x.data.shape[1:]).transpose((1, 2, 0))).save('result/preview/debug-{}.png'.format(name))


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


class CycleScalerUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.upscaler = kwargs.pop('upscaler')
        self.downscaler = kwargs.pop('downscaler')
        self.first = kwargs.pop('first')
        self.switching_interval = kwargs.pop('switching_interval')
        self.iter = 0
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

    def loss_gen(self, name, enc, x_out, x_in, y_fake, lam1=100, lam2=1/16):
        loss_rec = lam1*self.loss_func_rec_gen(x_in, x_out)
        loss_adv = lam2*self.loss_func_adv_gen(y_fake)
        loss = loss_rec + loss_adv
        chainer.report({'{}:loss_rec'.format(name): loss_rec}, enc)
        chainer.report({'{}:loss_adv'.format(name): loss_adv}, enc)
        return loss

    def loss_dis(self, name, dis, y_real, y_fake):
        L1 = self.loss_func_adv_dis_real(y_real)
        L2 = self.loss_func_adv_dis_fake(y_fake)
        loss = L1 + L2
        chainer.report({'{}:loss_real'.format(name): L1}, dis)
        chainer.report({'{}:loss_fake'.format(name): L2}, dis)
        return loss

    def take_next_batch(self):
        batch_a = self.get_iterator('main').next()
        batch_b = self.get_iterator('trainB').next()

        x_l = Variable(xp.asarray([b[0] for b in batch_a]).astype('f'))
        x_s_nn = Variable(xp.asarray([b[1] for b in batch_a]).astype('f'))
        x_s = Variable(xp.asarray(batch_b).astype('f'))
        return x_l, x_s_nn, x_s

    def update_downscaler(self):
        optimizer = self.get_optimizer('downscaler')
        
        xp = self.downscaler.xp
        x_l, x_s_nn, x_s = self.take_next_batch()
        
        self.upscaler.gen.fix_broken_batchnorm()
        with chainer.using_config('train', False):
            x_sl = self.upscaler.gen(x_s)

        x_sls = self.downscaler.gen(x_sl)
        y_sls = self.downscaler.dis(x_sl, x_sls)
        y_s = self.downscaler.dis(x_sl, x_s)

        x_ls = self.downscaler.gen(x_l)
        with chainer.using_config('train', False):
            x_lsl = self.upscaler.gen(x_ls)
        y_lsl = self.upscaler.dis(y_ls, y_lsl)
        y_l = self.upscaler.dis(y_ls, y_l)

        self.downscaler.gen.cleargrads()

        loss_gen_down = self.loss_gen('down', self.downscaler.gen, x_sls, x_s, y_sls)
        loss_gen_up = self.loss_gen('up', self.downscaler.gen, x_lsl, x_l, y_lsl)
        loss_gen = loss_gen_down + loss_gen_up
        loss_gen.backward()
        
        x_in.unchain_backward()
        x_out.unchain_backward()

        self.downscaler.dis.cleargrads()        
        loss_dis = self.loss_dis('down', self.downscaler.dis, y_s, y_sls)
        loss_dis.backward()

        self.upscaler.gen.cleargrads()
        self.upscaler.dis.cleargrads()
        optimizer.update()

    def update_upscaler(self):
        optimizer = self.get_optimizer('upscaler')
        
        xp = self.upscaler.xp
        x_l, x_s_nn, x_s = self.take_next_batch()
        
        self.downscaler.gen.fix_broken_batchnorm()
        with chainer.using_config('train', False):
            x_ls = self.downscaler.gen(x_l)

        x_lsl = self.upscaler.gen(x_ls)
        y_lsl = self.upscaler.dis(x_ls, x_lsl)
        y_l = self.upscaler.dis(x_ls, x_l)

        x_s_nn_l = self.upscaler.gen(x_s_nn)
        y_s_nn_l = self.upscaler.dis(x_s_nn, x_s_nn_l)

        self.upscaler.gen.cleargrads()
        loss_gen_up = self.loss_gen('up', self.upscaler.gen, x_lsl, x_ls, y_lsl)
        loss_gen_up_nn = self.loss_gen('up_nn', self.upscaler.gen, x_s_nn_l, x_s_nn, y_s_nn_l)
        loss_gen = loss_gen_up + loss_gen_up_nn
        loss_gen.backward()
        
        x_in.unchain_backward()
        x_out.unchain_backward()

        self.upscaler.dis.cleargrads()        
        loss_dis_up = self.loss_dis('up', self.downscaler.dis, y_l, y_lsl)
        loss_dis_up_nn = self.loss_dis('up_nn', self.downscaler.dis, y_l, y_s_nn_l)
        loss_dis = loss_dis_up + loss_dis_up_nn
        loss_dis.backward()

        self.downscaler.gen.cleargrads()
        self.downscaler.dis.cleargrads()
        optimizer.update()

    def update_core(self):
        switching_interval = self.switching_interval
        if self.first == 'down':
            update_first, update_second = self.update_downscaler, self.update_upscaler
        elif self.first == 'up':
            update_first, update_second = self.update_upscaler, self.update_downscaler
        else:
            assert False, 'unkwown mode {}'.foramt(self.first)
        if switching_interval is None:
            update_first()
        elif self.iter % (switching_interval * 2) < switching_interval:
            update_first()
        else:
            update_second()
        self.iter += 1


