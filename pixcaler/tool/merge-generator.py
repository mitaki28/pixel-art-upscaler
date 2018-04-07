import fire
import chainer
import pixcaler.net
from pathlib import Path

class GeneratorMerge(object):
    def __init__(self, clean=False, in_ch=4, out_ch=4, base_ch=64):
        self.clean = clean
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch

    def by_path(self, enc_path, dec_path, gen_path):
        enc_path = Path(enc_path)
        dec_path = Path(dec_path)
        gen_path = Path(gen_path)
        if gen_path.exists():
            print("{} is already exists".format(gen_path))
            exit(-1)
        gen = net.Generator(self.in_ch, self.out_ch, self.base_ch)
        chainer.serializers.load_npz(str(enc_path), gen.enc)
        chainer.serializers.load_npz(str(dec_path), gen.dec)
        chainer.serializers.save_npz(str(gen_path), gen)
        if clean:
            enc_path.unlink()
            dec_path.unlink()

    def by_iteration(self):
        model_dir = Path(model_dir)
        enc_path = model_dir/'enc_iter_{}.npz'.format(iteration)
        dec_path = model_dir/'dec_iter_{}.npz'.format(iteration)
        gen_path = model_dir/'gen_iter_{}.npz'.format(iteration)
        self.by_path(enc_path, dec_path, gen_path, in_ch, out_ch, base_ch)

if __name__ == '__main__':
    fire.Fire(GeneratorMerge)