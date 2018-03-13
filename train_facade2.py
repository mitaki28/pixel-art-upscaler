#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers

from net import Discriminator
from net import Encoder
from net import Decoder
from updater import FacadeUpdater2

from facade_dataset import FacadeDataset, HiResoDataset
from facade_visualizer import out_image2

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--batchsize', '-b', type=int, default=4,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='./image/fsm',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=10,
                        help='Interval of displaying log to console')
    parser.add_argument('--preview_interval', type=int, default=100,
                        help='Interval of previewing generated image')
    parser.add_argument('--encoder0', help='path to encoder base model')
    parser.add_argument('--decoder0', help='path to decoder base model')
    parser.add_argument('--discriminator0', help='path to discriminator base model')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    enc0 = Encoder(in_ch=4)
    dec0 = Decoder(out_ch=4)
    dis = Discriminator(in_ch=4, out_ch=4)
    chainer.serializers.load_npz(args.encoder0, enc0)
    chainer.serializers.load_npz(args.decoder0, dec0)
    chainer.serializers.load_npz(args.discriminator0, dis)

    enc = Encoder(in_ch=4)
    dec = Decoder(out_ch=4)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer


    # Setup an optimizer
    # def make_optimizer(model, alpha=0.0002, beta1=0.5):
    #     optimizer = chainer.optimizers.SGD(0.000002)
    #     optimizer.setup(model)
    #     optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
    #     return optimizer

    opt_enc = make_optimizer(enc)
    opt_dec = make_optimizer(dec)
    opt_dis = make_optimizer(dis)

    train_d = HiResoDataset(
        "{}/main/front".format(args.dataset),
    )
    test_d = HiResoDataset(
        "{}/test/front".format(args.dataset),        
    )
    # train_d = FacadeDataset(
    #     "{}/main/back".format(args.dataset),
    #     "{}/main/front".format(args.dataset),
    # )
    # test_d = FacadeDataset(
    #     "{}/test/back".format(args.dataset),
    #     "{}/test/front".format(args.dataset),        
    # )
    #train_iter = chainer.iterators.MultiprocessIterator(train_d, args.batchsize, n_processes=4)
    #test_iter = chainer.iterators.MultiprocessIterator(test_d, args.batchsize, n_processes=4)
    train_iter = chainer.iterators.SerialIterator(train_d, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_d, args.batchsize)

    # Set up a trainer
    updater = FacadeUpdater2(
        models=(enc0, dec0, enc, dec, dis),
        iterator={
            'main': train_iter,
            'test': test_iter,
        },
        optimizer={
            'enc': opt_enc,
            'dec': opt_dec, 
            'dis': opt_dis,
        },
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    preview_interval = (args.preview_interval, 'iteration')
    
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.iteration}.npz'),
                   trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        enc, 'enc_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dec, 'dec_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'enc/loss', 'dec/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_image2(
            updater, enc0, dec0, enc, dec,
            2, 2, args.seed, args.out),
        trigger=preview_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
