import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

class NNConvolution2D(chainer.Chain):

    def __init__(self, in_channels, out_channels, r, ksize=3, stride=1, padding=1, nobias=False, initialW=None, initial_bias=None):
        self.r = r
        super().__init__(
            conv=L.Convolution2D(in_channels, out_channels, ksize, stride, padding, nobias, initialW, initial_bias),
        )

    def __call__(self, x):
        return self.conv(F.unpooling_2d(x, self.r, self.r, 0, cover_all=False))

# https://gist.github.com/musyoku/849094afca2889d9024f59e683fa7036
class PixelShuffler(chainer.Chain):

    def __init__(self, in_channels, out_channels, r, nobias=False, initialW=None, initial_bias=None):
        self.r = r
        super().__init__(
            conv=L.Convolution2D(in_channels, (r ** 2) * out_channels, 3, 1, 1, nobias, initialW, initial_bias),
        )

    def __call__(self, x):
        r = self.r
        out = self.conv(x)
        batchsize = out.shape[0]
        in_channels = out.shape[1]
        out_channels = in_channels // (r ** 2)
        in_height = out.shape[2]
        in_width = out.shape[3]
        out_height = in_height * r
        out_width = in_width * r

        out = F.reshape(out, (batchsize, r, r, out_channels, in_height, in_width))
        out = F.transpose(out, (0, 3, 4, 1, 5, 2))
        out = F.reshape(out, (batchsize, out_channels, out_height, out_width))
        return out


# U-net https://arxiv.org/pdf/1611.07004v1.pdf

# convolution-batchnormalization-(dropout)-relu
class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        elif sample=='up':
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        elif sample=='up-nn':
            layers['c'] = NNConvolution2D(ch0, ch1, 2, 3, 1, 1, initialW=w)
        elif sample=='none':
            layers['c'] = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w)
        elif sample=='none-5':
            layers['c'] = L.Convolution2D(ch0, ch1, 5, 1, 2, initialW=w)
        else:
            assert False, 'unknown sample {}'.format(sample)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)
        
    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h

class Encoder(chainer.Chain):
    def __init__(self, in_ch, base_ch=64):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = L.Convolution2D(in_ch, base_ch, 5, 1, 2, initialW=w)
        layers['c1'] = CBR(base_ch, base_ch * 2, bn=True, sample='none', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(base_ch * 2, base_ch * 4, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(base_ch * 4, base_ch * 8, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(base_ch * 8, base_ch * 8, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(base_ch * 8, base_ch * 8, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c6'] = CBR(base_ch * 8, base_ch * 8, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c7'] = CBR(base_ch * 8, base_ch * 8, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        super().__init__(**layers)

    def __call__(self, x):
        hs = [F.leaky_relu(self.c0(x))]
        for i in range(1,8):
            hs.append(self['c%d'%i](hs[i-1]))
        return hs

class Decoder(chainer.Chain):
    def __init__(self, out_ch, base_ch=64):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = CBR(base_ch * 8, base_ch * 8, bn=True, sample='up-nn', activation=F.relu, dropout=True)
        layers['c1'] = CBR(base_ch * 8 + base_ch * 8, base_ch * 8, bn=True, sample='up-nn', activation=F.relu, dropout=True)
        layers['c2'] = CBR(base_ch * 8 + base_ch * 8, base_ch * 8, bn=True, sample='up-nn', activation=F.relu, dropout=True)
        layers['c3'] = CBR(base_ch * 8 + base_ch * 8, base_ch * 8, bn=True, sample='up-nn', activation=F.relu, dropout=False)
        layers['c4'] = CBR(base_ch * 8 + base_ch * 8, base_ch * 4, bn=True, sample='up-nn', activation=F.relu, dropout=False)
        layers['c5'] = CBR(base_ch * 4 + base_ch * 4, base_ch * 2, bn=True, sample='up-nn', activation=F.relu, dropout=False)
        layers['c6'] = CBR(base_ch * 2 + base_ch * 2, base_ch, bn=True, sample='none', activation=F.relu, dropout=False)
        layers['c7'] = L.Convolution2D(base_ch + base_ch, out_ch, 5, 1, 2, initialW=w)
        super().__init__(**layers)

    def __call__(self, hs):
        h = self.c0(hs[-1])
        for i in range(1,8):
            h = F.concat([h, hs[-i-1]])
            if i<7:
                h = self['c%d'%i](h)
            else:
                h = self.c7(h)
        return h

class Generator(chainer.Chain):
    def __init__(self, in_ch, out_ch, base_ch=64):        
        super().__init__(
            enc=Encoder(in_ch, base_ch),
            dec=Decoder(out_ch, base_ch),
        )

    def __call__(self, x_in):
        return self.dec(self.enc(x_in))

    def fix_broken_batchnorm(self):
        '''
        add-hook fix for broken batch normalization
        '''
        xp = self.xp
        for i in range(1, 8):
            cbr = self.enc['c{}'.format(i)]
            if xp.isnan(cbr.batchnorm.avg_var[-1]):
                cbr.batchnorm.avg_var = xp.zeros(cbr.batchnorm.avg_var.shape, cbr.batchnorm.avg_var.dtype)
        for i in range(0, 7):
            cbr = self.dec['c{}'.format(i)]
            if xp.isnan(cbr.batchnorm.avg_var[-1]):
                cbr.batchnorm.avg_var = xp.zeros(cbr.batchnorm.avg_var.shape, cbr.batchnorm.avg_var.dtype)        

class Discriminator(chainer.Chain):
    def __init__(self, in_ch, out_ch, flat=True, base_ch=64):
        assert base_ch % 2 == 0
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0_0'] = CBR(in_ch, base_ch // 2, bn=False, sample=('none-5' if flat else 'down'), activation=F.leaky_relu, dropout=False)
        layers['c0_1'] = CBR(out_ch, base_ch // 2, bn=False, sample=('none-5' if flat else 'down'), activation=F.leaky_relu, dropout=False)
        layers['c1'] = CBR(base_ch, base_ch * 2, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(base_ch * 2, base_ch * 4, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(base_ch * 4, base_ch * 8, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = L.Convolution2D(base_ch * 8, 1, 3, 1, 1, initialW=w)
        super().__init__(**layers)

    def __call__(self, x_0, x_1):
        h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        return h

class Pix2Pix(chainer.Chain):
    def __init__(self, in_ch, out_ch, flat=True, base_ch=64):
        super().__init__(
            gen=Generator(in_ch, out_ch, base_ch=base_ch),
            dis=Discriminator(in_ch, out_ch, base_ch=base_ch, flat=flat),
        )