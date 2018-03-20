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

from net import Encoder
from net import Decoder, DownscaleDecoder

from facade_visualizer import convert_image

from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('images', help='path to input images', metavar='image', type=str, nargs='*',)

    parser.add_argument('--input_dir', '-i')    
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', type=str, default='out/converted', help='path to output directory')
    parser.add_argument('--downscale', action='store_true', default=False, help='scale output image to half size or not')
    parser.add_argument('--compare', action='store_true', default=False, help='scale output image to half size or not')

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

    out_dir = Path(args.out)
    single_dir = out_dir/'single'
    compare_dir = out_dir/'compare'

    single_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir is not None:
        image_paths = Path(args.input_dir).glob('*.png')
    else:
        image_paths = [Path(image_path_str) for image_path_str in args.images]        
    for image_path in image_paths:
        single_path = single_dir/image_path.name
        compare_path = compare_dir/image_path.name
        with Image.open(image_path) as img:
            img = img.convert('RGBA')
            W, H = img.size
            converted_img = convert_image(img, enc, dec).convert('RGBA')
            converted_img.save(single_path)
            print(image_path, '->', single_path)

            if args.compare:
                compare_img = Image.new('RGBA', (2 * W, H))
                compare_img.paste(img, (0, 0))
                compare_img.paste(converted_img, (W, 0))
                compare_img.save(compare_path)
                print(image_path, '->', compare_path)
        

if __name__ == '__main__':
    main()
