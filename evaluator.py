"""eval_factorvae.py"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Lock
tqdm.set_lock(Lock())

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import model.model as model


'''
NOTE:
    1. Expected Dataset Tree

        dset_root
        |_______ vote1_[factor_index]_[factor_name]
                 |_______ 0.jpg
                 |_______ 1.jpg
                 |_______ ...

        |_______ vote2_[factor_index]_[factor_name]
                 |_______ 0.jpg
                 |_______ 1.jpg
                 |_______ ...

    2. when use this code on other networks, make sure that
        2-1. input pre-processings are set properly. (search HERE1)
        2-2. the network is initialized with proper checkpoint. (search HERE2)
        2-3. Encoder class keeps and wraps the inference network.
                e.g. enc = Encoder(my_network)
             When images are fed into the Encoder instance, it should output corresponding z.
                e.g. z = enc(x)

            (search HERE3)
'''


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Encoder(object):
    #### HERE3 ####
    def __init__(self, DR, G):
        self.DR = DR
        self.G = G

    def forward(self, x):
        z = self.DR(x)
        r = self.G(z)
        r = r.contiguous().view(r.size(0), r.size(1)).data
        return r

    def __call__(self, x):
        return self.forward(x)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if img.mode == 'L' or img.mode == 'LA':
            return img.convert('L')
        else:
            return img.convert('RGB')


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, loader=pil_loader):
        super(CustomImageFolder, self).__init__(root, transform, loader=loader)
        self.nv, self.nk, self.L = self.get_params()

    def get_params(self):
        dset_table = np.array(self.imgs)
        num_votes = len(self.classes)

        factors = [vote_info.split('_')[1] for vote_info in self.classes]
        factors = list(set(factors))
        num_factors = len(factors)

        Ls = []
        for vote in range(num_votes):
            L = (dset_table[:, 1] == str(vote)).sum()
            Ls.append(L)
        Ls = np.array(Ls)
        if Ls.mean() != Ls[0]:
            raise('the numbers of samples(L) in each vote are not equal to each other. mean L = {}.'.format(Ls.mean()))

        return num_votes, num_factors, Ls[0]

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        k = int(os.path.dirname(path).split('/')[-1].split('_')[1])
        k = torch.tensor(k).float()

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, k


def load_data(args):
    #### HERE1 ####
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # data for calculating empirical standard deviations
    dset = ImageFolder(root=args.dset_dir, transform=transform, loader=pil_loader)
    dloader = DataLoader(dset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=4,
                         pin_memory=True,
                         drop_last=False)

    # data for training majoirty-vote classifier
    train_dset = CustomImageFolder(root=args.dset_dir, transform=transform, loader=pil_loader)
    train_loader = DataLoader(train_dset,
                              batch_size=int(train_dset.L),
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=False)

    return dloader, train_loader


class Evaluator(object):
    def __init__(self, args, dloader):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'

        self.name = args.name
        self.ckpt_path = args.ckpt_path
        self.dloader, self.train_loader = dloader

        self.encoder, self.z_dim, self.global_iter = self.load_model()
        self.emp_std, self.emp_var = self.calc_empirical_std()
        self.collapsed = self.emp_var < args.cvth
        self.fill_collapsed = torch.ones_like(self.emp_var)*1e5

    def __call__(self, num_sample):
        return self.eval(num_sample)

    def load_model(self):
        #### HERE2 ####
        checkpoint = torch.load(self.ckpt_path.open('rb'))
        z_dim = checkpoint['z_dim'] #
        r_dim = checkpoint['r_dim'] #
        ngf = checkpoint['model_states']['ngf']
        ndf = checkpoint['model_states']['ndf']
        nc = checkpoint['model_states']['nc']
        DR = model.Discriminator(z_dim=z_dim, nc=nc, ndf=ndf, metric='factorvae').to(self.device)
        DR.load_state_dict(checkpoint['model_states']['DR']) #
        DR.eval()

        G = model.Generator(z_dim=z_dim, r_dim=r_dim, nc=nc, ngf=ngf, metric='factorvae').to(self.device)
        G.load_state_dict(checkpoint['model_states']['G']) #
        G.eval()

        E = Encoder(DR, G)
        global_iter = checkpoint['iter']
        print("=> loaded checkpoint '{} (iter {})'".format(self.ckpt_path, global_iter))

        return E, r_dim, global_iter #

    def calc_empirical_std(self):
        N = len(self.dloader.dataset)
        zs = []
        loader = iter(self.dloader)
        tqdm.write('now extract empirical standard deviation from entire dataset')
        for batch_idx in tqdm(range(len(self.dloader))):
            imgs, _ = loader.next()
            imgs = imgs.to(self.device)
            with torch.no_grad():
                z = self.encoder(imgs)

            zs.append(z)

        zs = torch.cat(zs)
        return zs.std(0), zs.var(0)

    def eval(self):
        if (self.collapsed == 1).sum().item() == self.z_dim:
            score = 0
            k_preds = ['x' for _ in range(self.z_dim)]
            return score, k_preds
        # make votes
        votes = []
        loader = iter(self.train_loader)
        tqdm.write('now training classifier')
        for _ in tqdm(range(len(loader))):
            img, k = loader.next()
            if k.mean() != k[0]:
                raise('data parsing error')

            img = img.to(self.device)
            k = int(k[0].item())

            z = self.encoder(img)
            normed_z = z.div(self.emp_std)
            var_normed_z = normed_z.var(0)
            d = torch.where(self.collapsed, self.fill_collapsed, var_normed_z).min(-1)[-1].item()
            vote = [d, k]
            votes.append(vote)
        votes = np.array(votes)

        # train classifier
        V = []
        nk = self.train_loader.dataset.nk
        for z in range(self.z_dim):
            V_j = []
            for k in range(nk):
                V_jk = (votes[np.where(votes[:, 0] == z)][:, 1] == k).sum()
                V_j.append(V_jk)
            V.append(V_j)

        V = np.array(V)
        score = V.max(1).sum()/V.sum()

        k_preds = []
        for k_pred, z_collapsed in zip(V.argmax(1), self.collapsed):
            if z_collapsed.item() == 1:
                k_pred = 'x'
            else:
                k_pred = str(k_pred)
            k_preds.append(k_pred)

        return score, k_preds


def main(args):
    results = dict(name=args.name)

    scores = dict()
    logiter = args.logiter
    lastiter = args.lastiter

    ckpt_dir = Path(args.ckpt_dir).joinpath(args.name)
    ckpts = [ckpt_dir.joinpath(str(it)) for it in range(logiter, lastiter+1, logiter)]
    dloader = load_data(args)

    out = False
    max_score = 0
    max_score_iter = 0
    count = 0
    for ckpt in ckpts:
        if not ckpt.is_file():
            continue

        args.ckpt_name = ckpt.name
        args.ckpt_path = ckpt
        evaluator = Evaluator(args, dloader)
        score, k_preds = evaluator.eval()
        if max_score < score:
            max_score = score
            max_score_iter = evaluator.global_iter

        result = dict()
        result['score'] = score
        result['classifier_decision'] = k_preds
        scores[int(evaluator.global_iter)] = result

        print('learned classifier')
        print(' '.join(str(dim) for dim in range(len(k_preds))))
        print(' '.join(k_preds))
        print('metric result:{:.4f}, best:{:.4f}(iter:{})'.format(score, max_score, max_score_iter))
        print()

        count += 1
        if count == len(ckpts):
            out = True

    results['scores'] = scores
    torch.save(results, ckpt_dir.joinpath('result.metric').open('wb+'))
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='metric evaluator proposed by KIM et al.')

    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--logiter', default=1000, type=int, help='')
    parser.add_argument('--lastiter', default=10000, type=int, help='')
    parser.add_argument('--ckpt_dir', default='checkpoint', type=str, help='checkpoint directory')
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--name', default='main', type=str, help='the name of the experiment')

    parser.add_argument('-bs', '--batch_size', default=1024, type=int, help='batch size')
    #parser.add_argument('-nk', '--num_factors', type=int, help='the number of ground truth generative factors')
    #parser.add_argument('-nv', '--num_votes', default=2000, type=int, help='the number of votes')
    #parser.add_argument('-ns', '--L', default=200, type=int, help='the number of samples per vote')
    parser.add_argument('-c', '--collapse', default=False, action='store_true', help='remove collapsed dimension')
    parser.add_argument('-cvth', default=0.05, type=float, help='any dimensions of the empirical variances below this value will be considered collapsed')

    args = parser.parse_args()
    main(args)
