import fire

import numpy as np
import net

import keras
import keras.layers
import chainer.serializers

from pathlib import Path
from PIL import Image

import subprocess

def get_weights_convolution(chainer_model):
    return [
        chainer_model.W.data.transpose((2, 3, 1, 0)),
        chainer_model.b.data,
    ]

def get_weights_bn(chainer_model):
    return [
        chainer_model.gamma.data,
        chainer_model.beta.data,
        chainer_model.avg_mean,
        np.zeros(chainer_model.avg_var.shape) if np.isnan(chainer_model.avg_var[-1]) else chainer_model.avg_var,
    ]

def chainer_bn(chainer_model):
    return keras.layers.normalization.BatchNormalization(
        momentum=0.9,
        epsilon=2e-05,
        weights=get_weights_bn(chainer_model),
    )

def chainer_leaky_relu():
    return keras.layers.advanced_activations.LeakyReLU(
        alpha=0.2,
    )

def down_cbr(filters, h, cbr_chainer):
    h = keras.layers.Conv2D(
        filters=filters,
        kernel_size=4,
        strides=2,
        padding='same',
        weights=get_weights_convolution(cbr_chainer.c),
    )(h)
    h = chainer_bn(cbr_chainer.batchnorm)(h)
    h = chainer_leaky_relu()(h)
    return h

def up_cbr(filters, dropout, h, cbr_chainer):
    h = keras.layers.UpSampling2D(
        size=(2, 2),
    )(h)
    h = keras.layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='same',
        weights=get_weights_convolution(cbr_chainer.c.conv),
    )(h)

    h = chainer_bn(cbr_chainer.batchnorm)(h)
    if dropout:
        h = keras.layers.Dropout(0.5)(h)
    h = keras.layers.core.Activation('relu')(h)
    return h

def generator(out_ch, input_keras, encoder_chainer, decoder_chainer):

    h = input_keras

    h = keras.layers.Conv2D(
        filters=64,
        kernel_size=5,
        strides=1,
        padding='same',
        weights=get_weights_convolution(encoder_chainer.c0),
    )(h)

    h0 = chainer_leaky_relu()(h)
    h = keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        padding='same',
        weights=get_weights_convolution(encoder_chainer.c1.c),
    )(h0)
    h = chainer_bn(encoder_chainer.c1.batchnorm)(h)
    h1 = chainer_leaky_relu()(h)

    h2 = down_cbr(256, h1, encoder_chainer.c2)
    h3 = down_cbr(512, h2, encoder_chainer.c3)
    h4 = down_cbr(512, h3, encoder_chainer.c4)
    h5 = down_cbr(512, h4, encoder_chainer.c5)
    h6 = down_cbr(512, h5, encoder_chainer.c6)
    h7 = down_cbr(512, h6, encoder_chainer.c7)

    h = up_cbr(512, True, h7, decoder_chainer.c0)

    h = keras.layers.concatenate([h, h6])
    h = up_cbr(512, True, h, decoder_chainer.c1)

    h = keras.layers.concatenate([h, h5])
    h = up_cbr(512, True, h, decoder_chainer.c2)

    h = keras.layers.concatenate([h, h4])
    h = up_cbr(512, False, h, decoder_chainer.c3)

    h = keras.layers.concatenate([h, h3])
    h = up_cbr(256, False, h, decoder_chainer.c4)

    h = keras.layers.concatenate([h, h2])
    h = up_cbr(128, False, h, decoder_chainer.c5)

    h = keras.layers.concatenate([h, h1])
    h = keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        weights=get_weights_convolution(decoder_chainer.c6.c),
    )(h)
    h = chainer_bn(decoder_chainer.c6.batchnorm)(h)
    h = keras.layers.Activation('relu')(h)

    h = keras.layers.concatenate([h, h0])
    h = keras.layers.Conv2D(
        filters=out_ch,
        kernel_size=5,
        strides=1,
        padding='same',
        weights=get_weights_convolution(decoder_chainer.c7),
    )(h)
    return h

def load_model(model_dir, i, in_w, in_ch, out_ch):
    model_dir = Path(model_dir)
    enc = net.Encoder(in_ch)
    dec = net.Decoder(out_ch)
    chainer.serializers.load_npz(model_dir/'enc_iter_{}.npz'.format(i), enc)
    chainer.serializers.load_npz(model_dir/'dec_iter_{}.npz'.format(i), dec)

    input_keras = keras.layers.Input(shape=(in_w, in_w, in_ch))
    model = keras.models.Model(input_keras, generator(out_ch, input_keras, enc, dec))
    return model

def test_run(model_dir, iteration, img_path):
    model = load_model(model_dir, iteration, 64, 4, 4)
    with Image.open(img_path) as img:
        x_in = np.asarray(img.convert('RGBA')).astype("f").transpose((2, 0, 1)) / 127.5 - 1.0
        x_in = x_in.reshape((1,) + x_in.shape)
        x_in = np.concatenate([x_in, x_in, x_in, x_in])

        x_out = model.predict(x_in.transpose((0, 2, 3, 1))).transpose(0, 3, 1, 2)
        print(x_out)
        x_out = x_out[0].transpose(1, 2, 0)
        x_out = np.asarray(np.clip(np.asarray(x_out) * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        H, W, C = x_out.shape
        if C==1:
            x_out = x_out.reshape((H, W))
        Image.fromarray(x_out).convert('RGBA').show()

def export(model_dir, iteration, kerasjs_encoder,
    w_in=64, in_ch=4, out_ch=4, output_dir=Path(__file__).parent/'frontend'/'public'/'model'/'keras'):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=False)
    model = load_model(model_dir, iteration, w_in, in_ch, in_ch)
    with open(output_dir/'model_arch.json', 'w') as f:
        f.write(model.to_json())
    model.save(output_dir/'model.h5')
    subprocess.check_call(['python', kerasjs_encoder, '-q', str(output_dir/'model.h5')])
    

if __name__ == '__main__':
    fire.Fire()
        
