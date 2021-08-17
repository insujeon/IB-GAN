"""main.py"""

import argparse

import numpy as np
import torch

from solver_visualize import Solver
from eval_eastwood import Encoder
from eval_eastwood import load_data, str2list
from dataset_review import return_data
from utils import str2bool
from glob import glob

def get_codes(encoder, images, batch_size, use_cuda):
    chunks = torch.split(torch.arange(0, images.shape[0]).int(), batch_size)
    codes = []
    for chunk in chunks:
        x = images[chunk.numpy()]
        if use_cuda:
            x = x.cuda()
        x = x.view(batch_size, 1, 64, 64)
        c = encoder(x)
        codes.append(c.detach().cpu())
    codes = torch.cat(codes, dim=0)
    return codes

def main(args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    #data = load_data(args)
    #data = return_data(args)
    #images = data

    net = Solver(args)
    args.max_iter = int(args.max_iter)
    #ckpts = [str(c) for c in range(args.ckptiter, args.max_iter+1, args.ckptiter) if c >= 100000]
    ckpts = args.ckpt_name
    codes = torch.randn(args.nb_samples, args.r_dim)
    for ckpt in ckpts:
        net.load_checkpoint(str(ckpt))
        net.DR.metric = 'factorvae'
        net.G.metric = 'factorvae'
        net.DR.eval()
        net.G.eval()
        #encoder = Encoder(net.DR, net.G)
        #indices = np.random.permutation(images.shape[0])[:args.nb_samples]
        #codes = get_codes(encoder, images[indices], args.batch_size, args.cuda)
        net.visualize(codes, 0.8, 0.2)
        #code = torch.zeros(1,10).cuda()
        #net.visualize(code, 0, 1, 0.1, -1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VEEGAN')

    # Optimization
    parser.add_argument('--max_iter', default=1e5, type=int, help='training iteration size')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--D_lr', default=1e-5, type=float, help='learning rate for the Discriminator')
    parser.add_argument('--G_lr', default=1e-4, type=float, help='learning rate for the Generator')
    parser.add_argument('--R_lr', default=1e-4, type=float, help='learning rate for the Reconstructor')
    parser.add_argument('--beta', default=1, type=float, help='')
    parser.add_argument('--alpha', default=1, type=float, help='')
    parser.add_argument('--pretrain', default=False, type=str2bool, help='')

    # Network
    parser.add_argument('--ngf', default=64, type=int, help='noise dimension')
    parser.add_argument('--ndf', default=64, type=int, help='noise dimension')
    parser.add_argument('--z_dim', default=100, type=int, help='noise dimension')
    parser.add_argument('--r_dim', default=10, type=int, help='bottleneck dimension')
    parser.add_argument('--load_ckpt', default=-1, type=int, help='load checkpoint')
    parser.add_argument('--ckpt_dir', default='checkpoint', type=str, help='checkpoint directory')

    # Dataset
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='celeba', type=str, help='CelebA, CIFAR10')
    parser.add_argument('--num_workers', default=4, type=int, help='num_workers')

    # Visualization
    parser.add_argument('--viz', default=True, type=str2bool, help='enable visdom')
    parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=int, help='visdom port')
    parser.add_argument('--output_dir', default='output', type=str, help='image output directory')
    parser.add_argument('--save_img', default=False, type=str2bool, help='save eval images')
    parser.add_argument('--sample_num', default=30, type=int, help='the number of samples for visual inspection')
    parser.add_argument('--logiter', default=1000, type=int, help='logging iteration')
    parser.add_argument('--ptriter', default=5000, type=int, help='print iteration')
    parser.add_argument('--ckptiter', default=2500, type=int, help='ckpt save iteration')
    parser.add_argument('--ckpt_name', default=[600000,500000], type=str2list, help='target checkpoints')
    parser.add_argument('--nb_samples', default=100, type=int, help='# of sample to visualize')

    # misc
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')

    args = parser.parse_args()

    main(args)
