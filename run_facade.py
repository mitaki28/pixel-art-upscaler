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
from net import Decoder
from updater import FacadeUpdater

from facade_dataset import FacadeDataset
from facade_visualizer import convert_image

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('images', help='path to input images', metavar='image', type=str, nargs='+',)
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', type=str, default='out/converted', help='path to output directory')
    parser.add_argument('--encoder', help='path to encoder model')
    parser.add_argument('--decoder', help='path to decoder model')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('')

    # Set up a neural network to train
    enc = Encoder(in_ch=4)
    dec = Decoder(out_ch=4)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
        dis.to_gpu()

    chainer.serializers.load_npz(args.encoder, enc)
    chainer.serializers.load_npz(args.decoder, dec)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    image_paths = [Path(image_path_str) for image_path_str in args.images]
    imgs = [Image.open(str(image_path)) for image_path in image_paths]

    converted_imgs = convert_image(imgs, enc, dec)
    for image_path, img in zip(image_paths, converted_imgs):
        img.convert('RGBA').save(out/image_path.name)
        

if __name__ == '__main__':
    main()
