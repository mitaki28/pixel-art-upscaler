#!/usr/bin/env python

from __future__ import print_function

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

from functions.nearest_neighbor import nearest_neighbor

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
        elif sample=='down-b':
            layers['c'] = L.Convolution2D(ch0, ch1, 5, 1, 2, initialW=w)
        elif sample=='up':
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            # layers['c'] = PixelShuffler(ch0, ch1, 2, initialW=w)
        elif sample=='up-b':
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
    def __init__(self, in_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c6'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c7'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        super(Encoder, self).__init__(**layers)

    def __call__(self, x):
        hs = [F.leaky_relu(self.c0(x))]
        for i in range(1,8):
            hs.append(self['c%d'%i](hs[i-1]))
        return hs

class Decoder(chainer.Chain):
    def __init__(self, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c1'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c2'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c3'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c4'] = CBR(1024, 256, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c5'] = CBR(512, 128, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c6'] = CBR(256, 64, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c7'] = L.Convolution2D(128, out_ch, 3, 1, 1, initialW=w)
        super(Decoder, self).__init__(**layers)

    def __call__(self, hs):
        h = self.c0(hs[-1])
        for i in range(1,8):
            h = F.concat([h, hs[-i-1]])
            if i<7:
                h = self['c%d'%i](h)
            else:
                h = self.c7(h)
        return h

class DownscaleDecoder(chainer.Chain):
    def __init__(self, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c1'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c2'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c3'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c4'] = CBR(1024, 256, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c5'] = L.Convolution2D(512, out_ch, 3, 1, 1, initialW=w)
        self.n_layers = len(layers)
        super().__init__(**layers)

    def __call__(self, hs):
        h = self.c0(hs[-1])
        for i in range(1, self.n_layers):
            h = F.concat([h, hs[-i-1]])
            h = self['c%d'%i](h)
        return nearest_neighbor(h, 4)

class Discriminator(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0_0'] = CBR(in_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c0_1'] = CBR(out_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = L.Convolution2D(512, 1, 3, 1, 1, initialW=w)
        super(Discriminator, self).__init__(**layers)

    def __call__(self, x_0, x_1):
        h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        #h = F.average_pooling_2d(h, h.data.shape[2], 1, 0)
        return h
