import argparse
from pathlib import Path

import numpy as np
import chainer
from pixcaler.net import Generator


def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument(
        '--out', '-o', type=str,
        help='path to output directory',
    )
    parser.add_argument(
        '--generator', type=str, required=True,
        help='path to generator model',
    )
    parser.add_argument(
        '--factor', type=int, default=2,
    )
    parser.add_argument(
        '--input_size', type=int,
    )    
    
    args = parser.parse_args()
    gen_path = Path(args.generator)
    gen = Generator(in_ch=4, out_ch=4, factor=args.factor)
    chainer.serializers.load_npz(gen_path, gen)
    gen.fix_broken_batchnorm()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_size = args.input_size if args.input_size is not None else args.factor * 32

    x = chainer.Variable(np.empty((1, 4, input_size, input_size), dtype=np.float32))
    with chainer.using_config('train', False), chainer.using_config('enable_back_prop', False):
        y = gen(x)

    import webdnn.frontend.chainer as webdnn_chainer
    graph = webdnn_chainer.ChainerConverter().convert([x], [y])
    from webdnn.backend import generate_descriptor
    exec_info = generate_descriptor("webgl", graph)
    exec_info.save(out_dir)

if __name__ == '__main__':
    main()
