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

class GeneratorVisualizer(keras.callbacks.Callback):
    def __init__(self, preview_iteration_interval, test_iterator, n, out_dir):
        self.test_iterator = test_iterator
        self.n = n
        self.iteration = 0
        self.out_dir = Path(out_dir)
        self.epoch = 0
        self.preview_iteration_interval = preview_iteration_interval

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    def on_batch_begin(self, batch, logs={}):
        self.iteration += 1

    def on_batch_end(self, step, logs={}):
        if self.iteration % self.preview_iteration_interval != 0:
            return
        step += 1

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
            x_out = self.model.get_layer('Generator').predict(x_in)
    
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
        preview_path = preview_dir/'image_{:0>8}_{:0>8}.png'.format(self.epoch, step)
        current_path = preview_dir/'image_cureent.png'
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        img = Image.fromarray(x).convert('RGBA')
        img.save(preview_path)
        img.save(current_path)


class Pix2PixCheckpoint(keras.callbacks.Callback):
    def __init__(self, out_dir, period=1):
        self.out_dir = Path(out_dir)
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if epoch % self.period == 0:
            self.model.get_layer('Generator').save_weights(
                str(self.out_dir/'gen_{epoch:05d}.h5'.format(epoch=epoch)),
            )
            self.model.get_layer('Discriminator').save_weights(
                str(self.out_dir/'dis_{epoch:05d}.h5'.format(epoch=epoch)),
            )

class Pix2Pix(object):
    def __init__(self,
        size=64,
        in_ch=4,
        out_ch=4,
        base_ch=64,
    ):
        self.size = 64
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.pix2pix = pixcaler.keras.model.pix2pix(size, in_ch, out_ch, base_ch)

    def _load_generator(self, generator):
        gen = self.pix2pix.get_layer('Generator')
        gen.load_weights(generator)
        return gen

    def export_generator(self, generator, out_path, tfjs=False, kerasjs=None):
        out_path = Path(out_path)
        if out_path.exists():
            raise RuntimeError('{} is already exists'.format(out_path))
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
        preview_iteration_interval=100,
        snapshot_epoch_interval=10,
        initial_epoch=0,
        out_dir='result/',
        generator=None,
        discriminator=None,
        composite=False,
    ):
        if generator is not None:
            self.pix2pix.get_layer('Generator').load_weights(generator)
        if discriminator is not None:
            self.pix2pix.get_layer('Discriminator').load_weights(discriminator)
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        with (out_dir/'args').open('w') as f:
            f.write(json.dumps(sys.argv, sort_keys=True, indent=4))
        dataset_dir = Path(dataset_dir)
        if composite:
            train_dataset = pixcaler.dataset.CompositeAutoUpscaleDataset(str(dataset_dir))
            test_dataset = pixcaler.dataset.CompositeAutoUpscaleDataset(str(dataset_dir))
        else:
            train_dataset = pixcaler.dataset.AutoUpscaleDataset(str(dataset_dir/'main'))    
            test_dataset = pixcaler.dataset.AutoUpscaleDataset(str(dataset_dir/'test'))
        
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
                        np.zeros((batch_size, self.size // 8, self.size // 8, 1)),
                        np.zeros((batch_size, self.size // 8, self.size // 8, 1)),
                    ],
                ]
        self.pix2pix.compile(
            keras.optimizers.Adam(
                lr=0.0002,
                beta_1=0.5,
                beta_2=0.999,
                epsilon=1e-8,
                decay=0.0,
                amsgrad=False,
            ),
            [
                pixcaler.keras.model.gen_loss_l1,
                pixcaler.keras.model.gen_loss_adv,
                pixcaler.keras.model.dis_loss_real,
                pixcaler.keras.model.dis_loss_fake,        
            ],
        )
        self.pix2pix.fit_generator(
            _dataset(),
            math.ceil(len(train_dataset) / batch_size),
            epochs=epochs,
            initial_epoch=initial_epoch,
            verbose=1,
            callbacks=[
                Pix2PixCheckpoint(
                    out_dir,
                    snapshot_epoch_interval,
                ),
                # keras.callbacks.ModelCheckpoint(
                #     str(out_dir/'model_{epoch:02d}.hdf5'),
                #     monitor='val_loss',
                #     verbose=0,
                #     save_best_only=False,
                #     save_weights_only=True,
                #     mode='auto',
                #     period=1,
                # ),
                keras.callbacks.TensorBoard(
                    log_dir=str(out_dir/'logs'),
                    histogram_freq=0,
                    batch_size=None,
                    write_graph=True,
                    write_grads=False,
                    write_images=False,
                    embeddings_freq=0,
                    embeddings_layer_names=None,
                    embeddings_metadata=None,
                ),
                GeneratorVisualizer(
                    preview_iteration_interval=preview_iteration_interval,
                    test_iterator=test_iterator,
                    n=10,
                    out_dir=out_dir,
                )
            ]
        )

if __name__ == '__main__':
    fire.Fire(Pix2Pix)