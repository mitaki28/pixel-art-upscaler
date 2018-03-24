from pathlib import Path
import net

import numpy as np
import fire

import chainer

from webdnn.frontend.chainer import ChainerConverter
from webdnn.backend import generate_descriptor


def export(model_dir, iteration, in_ch=4, out_ch=4, output_dir=Path(__file__).parent/'frontend'/'public'/'model'/'webdnn'):
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    enc = net.Encoder(in_ch)
    dec = net.Decoder(out_ch)
    chainer.serializers.load_npz(model_dir/'enc_iter_{}.npz'.format(iteration), enc)
    chainer.serializers.load_npz(model_dir/'dec_iter_{}.npz'.format(iteration), dec)

    # hack: add-hook fix for broken batch normalization
    for i in range(1, 8):
        cbr = enc['c{}'.format(i)]
        if np.isnan(cbr.batchnorm.avg_var[-1]):
            print('enc.c{} is broken'.format(i))
            cbr.batchnorm.avg_var = np.zeros(cbr.batchnorm.avg_var.shape, cbr.batchnorm.avg_var.dtype)
    for i in range(0, 7):
        cbr = dec['c{}'.format(i)]
        if np.isnan(cbr.batchnorm.avg_var[-1]):
            print('dec.c{} is broken'.format(i))
            cbr.batchnorm.avg_var = np.zeros(cbr.batchnorm.avg_var.shape, cbr.batchnorm.avg_var.dtype)


    x_in = chainer.Variable(np.zeros((1, in_ch, 64, 64), dtype=np.float32))
    x_out = dec(enc(x_in))
    graph = ChainerConverter().convert([x_in], [x_out])
    exec_info = generate_descriptor("webgl", graph)
    exec_info.save(output_dir)

if __name__ == '__main__':
    fire.Fire()