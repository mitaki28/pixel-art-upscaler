from pathlib import Path
import math
import os
import sys
import json
import subprocess

import numpy as np
from PIL import Image
import fire

import keras
import chainer

import pixcaler.keras.model
import pixcaler.dataset
import pixcaler.scaler
import pixcaler.util
import pixcaler.visualizer

class KerasConverter(pixcaler.scaler.Converter):
    def __init__(self, gen, input_size):
        self.gen = gen
        self.input_size = input_size

    def get_input_size(self):
        return self.input_size

    def __call__(self, imgs):
        x_in = np.asarray([pixcaler.util.img_to_hwc_array(img) for img in imgs])
        x_out = self.gen.predict(x_in)
        return [pixcaler.util.hwc_array_to_img(x) for x in x_out]

class GeneratorVisualizer:
    def __init__(self, gen, test_iterator, n, out_dir):
        self.gen = gen
        self.test_iterator = test_iterator
        self.n = n
        self.out_dir = Path(out_dir)

    def __call__(self, iteration):
        n_pattern = 3
        n_images = self.n * n_pattern

        rows = self.n
        cols = n_pattern
        
        ret = []
        
        for it in range(self.n):
            batch = self.test_iterator.next()
            x_in = np.asarray([b[0] for b in batch]).astype('f')
            x_real = np.asarray([b[1] for b in batch]).astype('f')

            x_in = x_in.transpose((0, 2, 3, 1))
            x_real = x_real.transpose((0, 2, 3, 1))
            x_out = self.gen.predict(x_in)
    
            ret.append(x_in[0])
            ret.append(x_real[0])
            ret.append(x_out[0])

        x = np.asarray(np.clip(np.asarray(ret) * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        _, H, W, C = x.shape
        x = x.reshape((rows, cols, H, W, C))
        x = x.transpose(0, 2, 1, 3, 4)
        if C==1:
            x = x.reshape((rows*H, cols*W))
        else:
            x = x.reshape((rows*H, cols*W, C))
        preview_dir = self.out_dir/'preview'
        preview_path = preview_dir/'image_{:0>8}.png'.format(iteration)
        current_path = preview_dir/'image_cureent.png'
        if not preview_dir.exists():
            preview_dir.mkdir(exist_ok=True, parents=True)
        img = Image.fromarray(x).convert('RGBA')
        img.save(preview_path)
        img.save(current_path)


class Pix2PixCheckpoint:
    def __init__(self, gen, dis, out_dir):
        self.gen = gen
        self.dis = dis
        self.out_dir = Path(out_dir)

    def __call__(self, iteration):
        self.gen.save_weights(
            str(self.out_dir/'gen_{:0>8}.h5'.format(iteration)),
        )
        self.dis.save_weights(
            str(self.out_dir/'dis_{:0>8}.h5'.format(iteration)),
        )

class Pix2PixLogger:
    def __init__(self, out_path):
        self.out_path = Path(out_path)
        self.count = 0
        self.loss = {}

    def _clear_accumulation(self):
        self.count = 0
        self.loss = {}

    def accumulate(self, loss):
        self.count += 1
        for k in loss:                
            self.loss[k] = self.loss.get(k, 0) + loss[k]

    def get_current(self, iteration=None):
        loss = {}
        if iteration is not None:
            loss['iteration'] = iteration
        for k in self.loss:
            loss[k] = self.loss[k] / self.count
        return loss

    def flush(self, iteration):
        loss = self.get_current(iteration)
        print(loss)

        with self.out_path.open(mode='a') as f:
            f.write(json.dumps(loss, sort_keys=True))
            f.write('\n')
        self._clear_accumulation()

class Pix2Pix(object):
    def __init__(self,
        size=None,
        in_ch=4,
        out_ch=4,
        base_ch=64,
        factor=2,
    ):
        self.size = size if size is not None else int(factor * 32)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.factor = factor
        self.gen, self.dis, self.gen_trainer, self.dis_trainer = pixcaler.keras.model.pix2pix(
            size,
            in_ch,
            out_ch,
            base_ch,
            factor,
        )

    def _load_generator(self, generator):
        gen = self.gen
        gen.load_weights(generator)
        return gen

    def export_generator(self, generator, out_path, tfjs=False, kerasjs=None):
        out_path = Path(out_path)
        if out_path.exists():
            raise RuntimeError('{} is already exists'.format(out_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gen = self._load_generator(generator)
        gen.save(str(out_path))
        if tfjs:
            import tensorflowjs as tfjs
            tfjs.converters.save_keras_model(gen, out_path.parent)
        if kerasjs is not None:
            subprocess.check_call(['python', kerasjs, '-q', str(out_path)])

    def train(self,
        dataset_dir,
        epochs=200,
        batch_size=4,
        log_interval=100,
        preview_interval=1000,
        full_preview_interval=10000,
        snapshot_interval=10000,
        initial_iteration=0,
        out_dir='result/',
        generator=None,
        discriminator=None,
        use_lsgan=False,
        lam1=100,
        lam2=1/16,
    ):
        if generator is not None:
            self.gen.load_weights(generator)
        if discriminator is not None:
            self.dis.load_weights(discriminator)
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        with (out_dir/'args').open('w') as f:
            f.write(json.dumps(sys.argv, sort_keys=True, indent=4))
        dataset_dir = Path(dataset_dir)
        train_dataset = pixcaler.dataset.CompositeAutoUpscaleDataset(
            str(dataset_dir),
            fine_size=self.size,
            factor=self.factor,
        )
        test_dataset = pixcaler.dataset.CompositeAutoUpscaleDataset(
            str(dataset_dir),
            fine_size=self.size,
            factor=self.factor,
        )
        
        train_iterator = chainer.iterators.SerialIterator(
            train_dataset,
            batch_size=batch_size,
        )
        test_iterator = chainer.iterators.SerialIterator(
            test_dataset,
            batch_size=1,
        )

        def _dataset():
            for i, batch in enumerate(train_iterator):
                x_in = np.asarray([b[0] for b in batch]).astype('f')
                x_real = np.asarray([b[1] for b in batch]).astype('f')
                x_in = x_in.transpose((0, 2, 3, 1))
                x_real = x_real.transpose((0, 2, 3, 1))            
                yield [
                    [
                        x_in,
                        x_real,
                    ],
                    [
                        x_real,
                        np.zeros((batch_size, self.size // 8, self.size // 8, 1)),
                    ],
                ], [
                    [
                        x_in,
                        x_real,
                    ],
                    [
                        np.zeros((batch_size, self.size // 8, self.size // 8, 1)),
                        np.zeros((batch_size, self.size // 8, self.size // 8, 1)),
                    ],                    
                ]

        if use_lsgan:
            print('use LSGan with lam1 = {}, lam2 = {}'.format(lam1, lam2))
            loss_fn = pixcaler.keras.model.LsGanLoss(lam1, lam2)
        else:
            print('use BCEGan with lam1 = {}, lam2 = {}'.format(lam1, lam2))
            loss_fn = pixcaler.keras.model.BceGanLoss(lam1, lam2)
        
        self.gen_trainer.compile(
            keras.optimizers.Adam(
                lr=0.0002,
                beta_1=0.5,
                beta_2=0.999,
                epsilon=1e-8,
                decay=0.0,
                amsgrad=False,
            ),
            [
                loss_fn.gen_rec,
                loss_fn.gen_adv,
            ],
        )
        self.dis_trainer.compile(
            keras.optimizers.Adam(
                lr=0.0002,
                beta_1=0.5,
                beta_2=0.999,
                epsilon=1e-8,
                decay=0.0,
                amsgrad=False,
            ),
            [
                loss_fn.dis_real,
                loss_fn.dis_fake,  
            ],
        )

        checkpoint = Pix2PixCheckpoint(
            self.gen,
            self.dis,
            out_dir,
        )
        visualizer = GeneratorVisualizer(
            gen=self.gen,
            test_iterator=test_iterator,
            n=10,
            out_dir=out_dir,
        )
        logger = Pix2PixLogger(out_dir/'logs')
        converter = KerasConverter(self.gen, self.size)
        scaler = pixcaler.scaler.Upscaler(
            converter,
            self.factor,
            batch_size,
        )
        full_visualizer = pixcaler.visualizer.ScalerVisualizer(
            scaler,
            dataset_dir/'test',
            out_dir,
        )
        for i, ((gen_x, gen_y), (dis_x, dis_y)) in enumerate(_dataset(), initial_iteration + 1):
            _, loss_gen_rec, loss_gen_adv = self.gen_trainer.train_on_batch(gen_x, gen_y)
            _, loss_dis_real, loss_dis_fake = self.dis_trainer.train_on_batch(dis_x, dis_y)
            logger.accumulate({
                'gen/rec': loss_gen_rec,
                'gen/adv': loss_gen_adv,
                'dis/real': loss_dis_real,
                'dis/fake': loss_dis_fake,
            })
            print(logger.get_current(i), end='\r')
            if i % log_interval == 0:
                logger.flush(i)
            if i % preview_interval == 0:
                visualizer(i)
            if i % full_preview_interval == 0:
                full_visualizer(i)
            if i % snapshot_interval == 0:
                checkpoint(i)

if __name__ == '__main__':
    fire.Fire(Pix2Pix)
