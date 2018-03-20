#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import argparse
import os
from PIL import Image
from pathlib import Path

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers

from net import Discriminator
from net import Encoder
from net import Decoder, DownscaleDecoder
from updater import FacadeUpdater

from facade_dataset import FacadeDataset
from facade_visualizer import convert_image

from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('images', help='path to input images', metavar='image', type=str, nargs='+',)
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', type=str, default='out/converted', help='path to output directory')
    parser.add_argument('--downscale', type=bool, default=False, help='scale output image to half size or not')

    parser.add_argument('--model_dir', help='path to model directory')
    parser.add_argument('--iter', type=int, help='iteration of load model')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    print('GPU: {}'.format(args.gpu))
    print('')

    # Set up a neural network to train
    enc = Encoder(in_ch=4)
    if args.downscale:
        dec = DownscaleDecoder(out_ch=4)
    else:
        dec = Decoder(out_ch=4)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
        dis.to_gpu()

    chainer.serializers.load_npz(model_dir/'enc_iter_{}.npz'.format(args.iter), enc)
    chainer.serializers.load_npz(model_dir/'dec_iter_{}.npz'.format(args.iter), dec)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    image_paths = [Path(image_path_str) for image_path_str in args.images]
    for image_path in image_paths:
        out_path = out/image_path.name
        with Image.open(image_path) as img:
            convert_image(img, enc, dec).convert('RGBA').save(out_path)
            print(image_path, '->', out_path)
        

if __name__ == '__main__':
    main()
