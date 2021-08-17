"""main.py"""

import argparse

import numpy as np
import torch

from solver2 import Solver
from utils import str2bool


def main(args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    net = Solver(args)
    net.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IB-GAN main for (CelebA, 3dChairs)')

    # Optimization
    parser.add_argument('--max_iter', default=1e5, type=int, help='training iteration size')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--optim', default='adam', type=str, help='')
    parser.add_argument('--D_lr', default=1e-5, type=float, help='learning rate for the Discriminator')
    parser.add_argument('--G_lr', default=1e-4, type=float, help='learning rate for the Generator')
    parser.add_argument('--R_lr', default=1e-4, type=float, help='learning rate for the Reconstructor')
    parser.add_argument('--beta', default=1, type=float, help='')
    parser.add_argument('--gamma', default=1, type=float, help='')
    parser.add_argument('--alpha', default=1, type=float, help='')
    parser.add_argument('--z_bias', default=0, type=float, help='')
    parser.add_argument('--pretrain', default=False, type=str2bool, help='')
    parser.add_argument('--label_smoothing', default=False, type=str2bool, help='label smoothing on GAN labels')
    parser.add_argument('--instance_noise_start', default=0., type=float, help='start stddev of instance noise')
    parser.add_argument('--instance_noise_end', default=0., type=float, help='end stddev of instance noise')

    # Network
    parser.add_argument('--ngf', default=64, type=int, help='noise dimension')
    parser.add_argument('--ndf', default=64, type=int, help='noise dimension')
    parser.add_argument('--z_dim', default=64, type=int, help='noise dimension')
    parser.add_argument('--r_dim', default=10, type=int, help='bottleneck dimension')
    parser.add_argument('--r_weight', default=0, type=int, help='weight switch')
    parser.add_argument('--load_ckpt', default=-1, type=int, help='load checkpoint')
    parser.add_argument('--ckpt_dir', default='checkpoint_celeba', type=str, help='checkpoint directory')

    # Dataset
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='celeba', type=str, help='CelebA, CIFAR10')
    parser.add_argument('--num_workers', default=4, type=int, help='num_workers')

    # Visualization
    parser.add_argument('--viz', default=True, type=str2bool, help='enable visdom')
    parser.add_argument('--viz_name', default='main_celeba', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=int, help='visdom port')
    parser.add_argument('--output_dir', default='output_celeba', type=str, help='image output directory')
    parser.add_argument('--save_img', default=False, type=str2bool, help='save eval images')
    parser.add_argument('--sample_num', default=100, type=int, help='the number of samples for visual inspection')
    parser.add_argument('--mi_sample_num', default=10000, type=int, help='')
    parser.add_argument('--logiter', default=500, type=int, help='logging iteration')
    parser.add_argument('--ptriter', default=20000, type=int, help='print iteration')
    parser.add_argument('--ckptiter', default=20000, type=int, help='ckpt save iteration')

    # misc
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--init_type', default='normal', type=str, choices=['normal', 'orthogonal', 'original'])

    args = parser.parse_args()

    main(args)
