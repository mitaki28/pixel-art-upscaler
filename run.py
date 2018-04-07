import argparse
import os
from PIL import Image
from pathlib import Path

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers

from net import Generator

from pathlib import Path

from util import chw_array_to_img, img_to_chw_array

def transparent_background(img):
    background_color = img.getpixel((0, 0))
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if img.getpixel((i, j)) == background_color:
                img.putpixel((i, j), (0, 0, 0, 0))
    return img

def pad_power_of_2(img, minimum_size=32):
    W, H = img.size
    s = minimum_size
    while s < W or s < H:
        s *= 2
    ret = Image.new('RGBA', (s, s))
    ret.paste(img, ((s - W) // 2, (s - H) // 2))
    return ret

def convert_image(img, gen):
    xp = gen.xp
    
    x = img_to_chw_array(img)
    x_in = Variable(x.reshape(1, *x.shape))
    x_out, _ = gen(x_in)

    return chw_array_to_img(chainer.cuda.to_cpu(x_out.data)[0])

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('images', help='path to input images', metavar='image', type=str, nargs='*',)

    parser.add_argument('--input_dir', '-i')    
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', type=str, default='out/converted', help='path to output directory')
    parser.add_argument('--compare', action='store_true', default=False, help='scale output image to half size or not')

    parser.add_argument('--model', help='path to generator model')
    parser.add_argument('--downscale', action='store_true', default=False,
                        help='enable downscale learning',
    )
    
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    print('GPU: {}'.format(args.gpu))
    print('')

    gen = Generator(in_ch=4, out_ch=4)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()

    chainer.serializers.load_npz(model_path, gen)
    gen.fix_broken_batchnorm()

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
            oW, oH = img.size
            img = img.convert('RGBA')
            img = transparent_background(img)
            img = pad_power_of_2(img)
            if args.downscale:
                preprocessed_img = img
            else:
                preprocessed_img = img.resize((img.size[0] * 2, img.size[1] * 2), Image.NEAREST)
            
            converted_img = convert_image(preprocessed_img, gen).convert('RGBA')
            
            if args.downscale:
                cW, cH = converted_img.size
                pW, pH = (cW - oW, cH - oH)
                postprocessed_img = converted_img.crop((
                    pW // 2,
                    pH // 2,
                    pW // 2 + oW,
                    pH // 2 + oH,
                ))                
            else:
                cW, cH = converted_img.size
                pW, pH = (cW - oW * 2, cH - oH * 2)
                postprocessed_img = converted_img.crop((
                    pW // 2,
                    pH // 2,
                    pW // 2 + oW * 2,
                    pH // 2 + oH * 2
                ))
            postprocessed_img.save(single_path)
            print(image_path, '->', single_path)

            if args.compare:
                compare_img = Image.new('RGBA', (2 * cW, cH))
                compare_img.paste(preprocessed_img, (0, 0))
                compare_img.paste(converted_img, (cW, 0))
                compare_img.save(compare_path)
                print(image_path, '->', compare_path)
        

if __name__ == '__main__':
    main()
