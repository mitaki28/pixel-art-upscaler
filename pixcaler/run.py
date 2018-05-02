import argparse
from PIL import Image
from pathlib import Path

import chainer
from pixcaler.net import Generator
from pixcaler.scaler import Upscaler, Downscaler, ChainerConverter, Refiner


def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument(
        'images', metavar='image', type=str, nargs='*',
        help='path to input images', 
    )
    parser.add_argument(
        '--input_dir', '-i', type=str,
        help='directory containing input images (all png images in the directory are converted)'
    )
    parser.add_argument(
        '--gpu', '-g', type=int, default=-1,
        help='GPU ID (negative value indicates CPU)',
    )
    parser.add_argument(
        '--out', '-o', type=str, default='out/image/converted',
        help='path to output directory',
    )
    parser.add_argument(
        '--compare', action='store_true', default=False,
        help='output images for compare',
    )
    parser.add_argument(
        '--patch_size', '-p', type=int,
    )
    parser.add_argument(
        '--batch_size', '-b', type=int, default=1,
    )
    parser.add_argument(
        '--generator', type=str, required=True,
        help='path to generator model',
    )
    parser.add_argument(
        '--mode', type=str, choices=('up', 'down', 'refine'), default='up',
        help='scaling mode',
    )
    parser.add_argument(
        '--factor', type=int, default=2,
    )    
    
    args = parser.parse_args()
    gen_path = Path(args.generator)
    print('GPU: {}'.format(args.gpu))
    print('')

    gen = Generator(in_ch=4, out_ch=4, factor=args.factor)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()

    chainer.serializers.load_npz(gen_path, gen)
    gen.fix_broken_batchnorm()

    out_dir = Path(args.out)
    if args.compare:
        single_dir = out_dir/'single'
        compare_dir = out_dir/'compare'
        single_dir.mkdir(parents=True, exist_ok=True)
        compare_dir.mkdir(parents=True, exist_ok=True)
    else:
        single_dir = out_dir
        single_dir.mkdir(parents=True, exist_ok=True)

    class Logger:
        def __init__(self, context=""):
            self.context = context
        def on_patch(self, patch, idx, n):
            print("{}: {}/{}".format(self.context, idx + 1, n), end='\r')
    
    patch_size = args.patch_size if args.patch_size is not None else 32 * args.factor

    converter = ChainerConverter(gen, input_size=patch_size)
    logger = Logger()

    if args.mode == 'up':
        scaler = Upscaler(converter, factor=args.factor, batch_size=args.batch_size, handler=logger)
    elif args.mode == 'down':
        scaler = Downscaler(converter, factor=args.factor, batch_size=args.batch_size, handler=logger)
    elif args.mode == 'refine':
        scaler = Refiner(converter, batch_size=args.batch_size, handler=logger)    
    else:
        raise RuntimeError("unknown mode: {}".format(args.mode))

    if args.input_dir is not None:
        image_paths = Path(args.input_dir).glob('*.png')
    else:
        image_paths = [Path(image_path_str) for image_path_str in args.images]        
    for image_path in image_paths:
        logger.context = str(image_path)
        single_path = single_dir/image_path.name
        with Image.open(image_path) as img:
            img = img.convert('RGBA')
            converted_img = scaler(img)
            converted_img.save(single_path)
            print(image_path, '->', single_path)
            if args.compare:
                compare_path = compare_dir/image_path.name
                compareable_image = scaler.generate_comparable_image(img)
                w_comp, h_comp = compareable_image.size
                w_conv, h_conv = converted_img.size
                w_gen = w_comp + w_conv
                h_gen = max(h_comp, h_conv)
                compare_img = Image.new('RGBA', (w_gen, h_gen))
                compare_img.paste(compareable_image, (0, 0))
                compare_img.paste(converted_img, (w_comp, 0))
                compare_img.save(compare_path)
                print(image_path, '->', compare_path)
        

if __name__ == '__main__':
    main()
