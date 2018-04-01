from pathlib import Path

import numpy as np
from PIL import Image
import fire
import math
import os

import keras
import keras_model
import dataset
import chainer

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

def train(
    dataset_dir,
    epochs=200,
    size=64,
    in_ch=4,
    out_ch=4,
    batch_size=4,
    preview_iteration_interval=100,
    snapshot_epoch_interval=1,
    checkpoint=None,
    initial_epoch=0,
    out_dir='result/'
):
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    train_dataset = dataset.AutoUpscaleDataset(str(dataset_dir/'main'))
    train_iterator = chainer.iterators.SerialIterator(
        train_dataset,
        batch_size=batch_size,
    )
    
    test_iterator = chainer.iterators.SerialIterator(
        dataset.AutoUpscaleDataset(str(dataset_dir/'test')),
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
                    np.zeros((batch_size, size // 8, size // 8, 1)),
                    np.zeros((batch_size, size // 8, size // 8, 1)),
                    np.zeros((batch_size, size // 8, size // 8, 1)),
                ],
            ]
    pix2pix = keras_model.pix2pix(size, in_ch, out_ch)
    if checkpoint is not None:
        pix2pix.load_weights(checkpoint)
    pix2pix.compile(
        keras.optimizers.Adam(
            lr=0.0002,
            beta_1=0.5,
            beta_2=0.999,
            epsilon=1e-8,
            decay=0.0,
            amsgrad=False,
        ),
        [
            keras_model.gen_loss_l1,
            keras_model.gen_loss_adv,
            keras_model.dis_loss_real,
            keras_model.dis_loss_fake,        
        ],
    )
    pix2pix.fit_generator(
        _dataset(),
        math.ceil(len(train_dataset) / batch_size),
        epochs=epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(out_dir/'model_{epoch:02d}.hdf5'),
                monitor='val_loss',
                verbose=0,
                save_best_only=False,
                save_weights_only=True,
                mode='auto',
                period=1,
            ),
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
    fire.Fire()