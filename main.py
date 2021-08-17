"""main.py"""

import argparse
import random

import numpy as np
import torch

from solver import Solver
from utils import str2bool


def main(args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    
    net = Solver(args)
    net.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IB-GAN for (dsprites, cdsprites)')

    # Optimization
    parser.add_argument('--max_iter', default=5e5, type=int, help='training iteration size')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--optim', default='rmsprop', type=str, help='')
    parser.add_argument('--D_lr', default=1e-6, type=float, help='learning rate for the Discriminator')
    parser.add_argument('--G_lr', default=5e-5, type=float, help='learning rate for the Generator')
    parser.add_argument('--R_lr', default=5e-5, type=float, help='learning rate for the Reconstructor')
    parser.add_argument('--beta', default=0.071, type=float, help='optimal beta for color-dsprites')
    parser.add_argument('--gamma', default=1, type=float, help='optimal gamma for color-dsprites dataset')
    
    parser.add_argument('--alpha', default=1, type=float, help='deprecated') # deprecated paraemter
    parser.add_argument('--z_bias', default=0, type=float, help='deprecated') # deprecated pearemter 
    parser.add_argument('--pretrain', default=False, type=str2bool, help='deprecated') # deprecated paraemter 

    # Network
    parser.add_argument('--ngf', default=16, type=int, help='deprecated, please change model parameter ngf in solver_review.py') # deprecated paraemter
    parser.add_argument('--ndf', default=16, type=int, help='deprecated, please change model parameter ndf in solver_review.py') # deprecated parameter
    parser.add_argument('--z_dim', default=16, type=int, help='optimal z dimension for color-dsprites')
    parser.add_argument('--r_dim', default=10, type=int, help='optimal r dimension for color-dsprites')
    parser.add_argument('--load_ckpt', default=-1, type=int, help='load checkpoint')
    parser.add_argument('--ckpt_dir', default='checkpoint', type=str, help='checkpoint directory')

    # Dataset
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='cdsprites', type=str, help='CelebA, CIFAR10, dsprites')
    parser.add_argument('--num_workers', default=4, type=int, help='num_workers for dataloader')

    # Visualization
    parser.add_argument('--viz', default=True, type=str2bool, help='enable visdom')
    parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=int, help='visdom port')
    parser.add_argument('--output_dir', default='output', type=str, help='image output directory')
    parser.add_argument('--save_img', default=False, type=str2bool, help='save eval images')
    parser.add_argument('--sample_num', default=100, type=int, help='number of samples for the visualization')
    parser.add_argument('--mi_sample_num', default=10000, type=int, help='mutual inforamtion computation period')
    parser.add_argument('--logiter', default=500, type=int, help='console - logging iteration')
    parser.add_argument('--ptriter', default=2500, type=int, help='print iteration')
    parser.add_argument('--ckptiter', default=2500, type=int, help='ckpt save iteration')

    # misc
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable GPU usage')
    parser.add_argument('--init_type', default='normal', type=str, choices=['normal', 'orthogonal', 'original'])

    args = parser.parse_args()

    main(args)
