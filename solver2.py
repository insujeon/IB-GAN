from pathlib import Path

import visdom
import numpy as np

from PIL import Image

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from torchvision import transforms

from utils import cuda
import model.model2 as model
from dataset import return_data
import math

from tqdm import tqdm

win_samples = None
win_traverse = None

def noisy(image_tensor, running_sigma):
    """Add zero mean gaussian noise to input_tensor if running_sigma(standard deviation) > 0"""
    if (running_sigma <= 0)[0]:
        return image_tensor

    else:
        noise = Variable(image_tensor.data.new(image_tensor.size()).normal_()*running_sigma)
        noisy_x = torch.clamp(image_tensor + noise, -1, 1)
        return noisy_x

def runner(start_val, end_val, start_iter, end_iter, curr_iter):
    runval = ((end_val-start_val)/(end_iter-start_iter)*(curr_iter-start_iter) + start_val)
    return runval.clamp(0, start_val.item())


def kl_divergence(mu, logvar):
    kld_matrix = -0.5*(1 + logvar - mu**2 - logvar.exp())
    kld = kld_matrix.sum(1).mean()
    dimension_wise_kld = kld_matrix.mean(0)
    return kld, dimension_wise_kld

def kl_divergence3(mu, logvar, weight=1):
    kld_matrix = -0.5*(1 + logvar - mu**2 - logvar.exp())
    weighted_kld = kld_matrix * weight * mu.size(1)
    kld = weighted_kld.sum(1).mean()
    #kld = kld_matrix.sum(1).mean()
    dimension_wise_kld = kld_matrix.mean(0)
    return kld, dimension_wise_kld

def upper_mi(mu, logvar):
    kld_matrix = -0.5*(1 + logvar - mu**2 - logvar.exp())
    dimension_wise_kld = kld_matrix.sum(0)
    kld = dimension_wise_kld.sum()
    return kld, dimension_wise_kld

class Solver(object):
    def __init__(self, args):
        # misc
        self.args = args
        self.use_cuda = args.cuda and torch.cuda.is_available()

        # Optimization
        self.global_iter = 0
        self.max_iter = args.max_iter
        self.batch_size = args.batch_size
        self.D_lr = args.D_lr
        self.G_lr = args.G_lr
        self.R_lr = args.R_lr
        self.beta = args.beta
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.z_bias = args.z_bias
        self.r_weight = args.r_weight
        self.label_smoothing = args.label_smoothing
        self.instance_noise_start = args.instance_noise_start
        self.start_sigma = cuda(torch.FloatTensor([self.instance_noise_start]), self.use_cuda)
        self.instance_noise_end = args.instance_noise_end
        self.end_sigma = cuda(torch.FloatTensor([self.instance_noise_end]), self.use_cuda)

        # Dataset
        self.dataset = args.dataset
        self.data_loader = return_data(args)

        # Network
        self.ngf = args.ngf
        self.ndf = args.ndf
        self.z_dim = args.z_dim
        self.r_dim = args.r_dim
        self.fixed_r = self.sample_z(args.sample_num, dim=self.r_dim)
        self.fixed_r = Variable(cuda(self.fixed_r, self.use_cuda))
        self.fixed_z = self.sample_z(args.sample_num, dim=self.z_dim)
        self.fixed_z = Variable(cuda(self.fixed_z, self.use_cuda))
        self.load_ckpt = args.load_ckpt
        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.viz_name)
        self.win = None
        self.mi_win = None
        self.g_win = None
        self.main_win = None
        self.main_mi_win = None
        self.main_g_win = None
        self.model_init()

        self.stat_dict = dict()
        self.accurate_stat_dict = dict()

        # Visualization
        self.viz = args.viz
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.sample_num = args.sample_num
        self.mi_sample_num = args.mi_sample_num
        self.save_img = args.save_img
        self.output_dir = Path(args.output_dir).joinpath(args.viz_name)
        self.visualization_init()

        self.logiter = args.logiter
        self.ptriter = args.ptriter
        self.ckptiter = args.ckptiter
        self.pbar = tqdm(total=self.max_iter)

    def visualization_init(self):
        if self.viz:
            self.viz = visdom.Visdom(env='main', port=self.viz_port)

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def model_init(self):
        # different (hyper-)parameters for each dataset
        if self.dataset.lower() == 'dsprites':
            self.nc = 1
            self.ndf = 16
            self.ngf = 16
            decay = 1e-2
            DR = model.Discriminator
            Generator = model.Generator
        elif self.dataset.lower() == 'celeba':
            self.nc = 3
            """
            self.ndf = 64
            self.ngf = 64
            """
            decay = 0
            DR = model.Discriminator
            Generator = model.Generator
        elif self.dataset.lower() == '3dchairs':
            self.nc = 1
            self.ndf = 16
            self.ngf = 32
            decay = 1e-2
            DR = model.Discriminator
            Generator = model.Generator

        # init models
        self.G = cuda(Generator(self.z_dim, self.r_dim, self.nc, self.ngf, metric=None), self.use_cuda)
        self.DR = cuda(DR(self.z_dim, self.nc, self.ndf, metric=None), self.use_cuda)
        self.nets = [self.DR, self.G]
        for net in self.nets:
            if self.args.init_type == 'normal':
                net.apply(model.weights_init_normal)
            elif self.args.init_type == 'orthogonal':
                net.apply(model.weights_init_orthogonal)
            elif self.args.init_type == 'original':
                net.weight_init(mean=1.0, std=0.002)

        # get model params and init optimizers
        D_params = list(self.DR._modules['front'].parameters()) + \
                   list(self.DR._modules['disc'].parameters())
        GR_params = list(self.G._modules['fe'].parameters()) + \
                    list(self.G._modules['conv'].parameters()) + \
                    list(self.DR._modules['encoder'].parameters())
        G_emb_params = list(self.G._modules['emb'].parameters())

        opt_name = self.args.optim
        if opt_name == 'adam':
            self.D_optim = optim.Adam(D_params, lr=self.D_lr, betas=(0.9, 0.999))
            self.GR_optim = optim.Adam([{'params':G_emb_params}, {'params':GR_params}],
                                        lr=self.G_lr, betas=(0.9, 0.999))
        elif opt_name == 'sgd':
            self.D_optim = optim.SGD(D_params, lr=self.D_lr, momentum=0.9)
            self.GR_optim = optim.SGD([{'params':G_emb_params}, {'params':GR_params}],
                                        lr=self.G_lr, momentum=0.9)
        elif opt_name == 'rmsprop':
            self.D_optim = optim.RMSprop(D_params, lr=self.D_lr, momentum=0.9)
            self.GR_optim = optim.RMSprop([{'params':G_emb_params}, {'params':GR_params}],
                                        lr=self.G_lr, momentum=0.9)
        elif opt_name == 'adagrad':
            self.D_optim = optim.Adagrad(D_params, lr=self.D_lr)
            self.GR_optim = optim.Adagrad([{'params':G_emb_params}, {'params':GR_params}],
                                        lr=self.G_lr)
        elif opt_name == 'adadelta':
            self.D_optim = optim.Adadelta(D_params, lr=self.D_lr)
            self.GR_optim = optim.Adadelta([{'params':G_emb_params}, {'params':GR_params}],
                                        lr=self.G_lr)


        # make ckpt directory
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True)

        # load ckpt and continue training if necessary
        if self.load_ckpt != -1:
            self.load_checkpoint(str(self.load_ckpt))

    def calc_mis(self, running_sigma, chunk_size=10):
        self.set_mode('eval')
        H = 0
        D = 0
        lower = 0
        upper = 0
        indep = 0
        ntotal = 0
        inter = self.mi_sample_num // chunk_size
        batch_sizes = []
        while True:
            diff = self.mi_sample_num - ntotal
            if diff <= 0:
                break
            elif diff > inter:
                ntotal += inter
                batch_sizes.append(inter)
            else:
                ntotal += diff
                batch_sizes.append(diff)

        for batch_size in batch_sizes:
            # feedforward
            z = Variable(cuda(self.sample_z(batch_size=batch_size), self.use_cuda), requires_grad=False) + self.z_bias
            r_mu, r_logvar, r, x_g = self.G(z)
            noisy_x_g = noisy(x_g, running_sigma)
            z_recon, D_x_g = self.DR(noisy_x_g)

            obj_G = D_x_g.sum().item()
            obj_H = 0.5*( (z-self.z_bias).pow(2) + math.log(2*math.pi) ).sum().item()
            obj_D = 0.5*( (z-z_recon).pow(2) + math.log(2*math.pi) ).sum().item()
            obj_recon = obj_H-obj_D
            #obj_recon = 0.5*( (z-self.z_bias).pow(2) - (z-z_recon).pow(2)).sum().item()
            obj_kld, dim_kld = upper_mi(r_mu, r_logvar)

            G += obj_G/self.mi_sample_num
            H += obj_H/self.mi_sample_num
            D += obj_D/self.mi_sample_num
            lower += obj_recon/self.mi_sample_num
            upper += obj_kld.item()/self.mi_sample_num
            indep += dim_kld.data.cpu()/self.mi_sample_num

        self.set_mode('train')
        out = [lower, upper, *(indep.tolist()), H, D, G]
        return out

    def train(self):
        ones = Variable(cuda(torch.ones(self.batch_size, 1), self.use_cuda))
        zeros = Variable(cuda(torch.zeros(self.batch_size, 1), self.use_cuda))

        #sigma = cuda(torch.FloatTensor([sigma]), self.use_cuda)

        #sigma_stop_iter = cuda(torch.FloatTensor([15000]), self.use_cuda)
        #sigma_stop_iter = cuda(torch.FloatTensor([self.max_iter]), self.use_cuda)
        ##sigma_stop_iter = cuda(torch.FloatTensor([self.max_iter/1.5]), self.use_cuda)
        #sigma_stop_iter = cuda(torch.FloatTensor([int(self.max_iter*0.4)]), self.use_cuda)
        #sigma_stop_iter = cuda(torch.FloatTensor([0]), self.use_cuda)
        #sigma_stop_iter = torch.clamp(sigma_stop_iter, max=50000)

        dim_klds = []
        obj_klds = []
        obj_recons = []
        klds_iters = []
        obj_Gs = []

        if self.r_weight == 1 and self.r_dim == 10:
            weight = [0.01818182, 0.03636364, 0.05454545, 0.07272727, 0.09090909, 0.10909091, 0.12727273, 0.14545455, 0.16363636, 0.18181818]
            weight = torch.Tensor(weight).cuda().view(1,10)
            weight = weight.expand(self.batch_size,10)
        elif self.r_weight == 1 and self.r_dim == 15:
            weight = [0.008333333, 0.016666667, 0.025000000, 0.033333333, 0.041666667, 0.050000000, 0.058333333, 0.066666667, 0.075000000, 0.083333333, 0.091666667, 0.100000000, 0.108333333, 0.116666667, 0.125000000]
            weight = torch.Tensor(weight).cuda().view(1,15)
            weight = weight.expand(self.batch_size,15)
        else:
            weight = torch.ones(self.batch_size, self.r_dim).cuda()/self.r_dim

        out = False
        while not out:
            for (x, x2) in self.data_loader['train']:
                self.set_mode('train')
                self.global_iter += 1
                self.pbar.update(1)

                #running_sigma = torch.clamp((-sigma/sigma_stop_iter*self.global_iter + sigma), 0, sigma[0])
                running_sigma = runner(self.start_sigma, self.end_sigma, 0, self.max_iter, self.global_iter)

                # feedforward
                z = Variable(cuda(self.sample_z(), self.use_cuda)) + self.z_bias
                r_mu, r_logvar, r, x_g = self.G(z)
                noisy_x_g = noisy(x_g, running_sigma)
                z_recon, D_x_g = self.DR(noisy_x_g)

                x = Variable(cuda(x, self.use_cuda))
                noisy_x = noisy(x, running_sigma)
                z_g, D_x = self.DR(noisy_x)

                if self.label_smoothing:
                    ones_fake = torch.rand_like(ones) * 0.5 + 0.7
                else:
                    ones_fake = ones
                obj_D = F.binary_cross_entropy_with_logits(D_x_g, zeros) + \
                        F.binary_cross_entropy_with_logits(D_x, ones_fake)
                obj_G = F.binary_cross_entropy_with_logits(D_x_g, ones)
                #obj_G = D_x_g.mean()

                #obj_recon = ((self.z_dim-1) * math.log(2*math.pi) - F.mse_loss(z_recon, z))*0.5
                #obj_recon = ((self.z_dim-1) * math.log(2*math.pi)-(z_recon-z).pow(2).sum()/self.batch_size)*0.5
                obj_recon = 0.5*( (z-self.z_bias).pow(2) - (z-z_recon).pow(2)).sum(1).mean()

                #obj_H = 0.5*( (z-self.z_bias).pow(2) + math.log(2*math.pi) ).sum(1).mean()
                #obj_D = 0.5*( (z-z_recon).pow(2) + math.log(2*math.pi) ).sum(1).mean()

                obj_kld, dim_kld = kl_divergence(r_mu, r_logvar)

                #obj_GR = self.alpha * obj_G + -obj_recon + self.beta * obj_kld
                obj_GR = self.gamma * obj_G + self.alpha * (-obj_recon) + self.beta * obj_kld

                # backward & update network params
                self.D_optim.zero_grad()
                obj_D.backward(retain_graph=True)
                self.D_optim.step()

                self.GR_optim.zero_grad()
                obj_GR.backward()
                self.GR_optim.step()

                # print, visualize, etc
                if self.global_iter%self.logiter == 0:
                    dim_klds.append(dim_kld.data)
                    obj_klds.append(obj_kld.data)
                    obj_recons.append(obj_recon.data)
                    klds_iters.append(self.global_iter)
                    obj_Gs.append(obj_G.data)
                    self.stat_dict[self.global_iter] = [obj_kld.item(), obj_recon.item(), obj_G.item()]

                    self.pbar.write(' ')
                    self.pbar.write('{} {}'.format(self.global_iter, self.viz_name))
                    #self.pbar.write('{} {}'.format(obj_recon, obj_H-obj_D))
                    self.pbar.write('obj_D:{:.3f} obj_G:{:.3f}, obj_recon:{:.3f}, obj_kld:{:.3f}'.
                          format(obj_D.item(), obj_G.item(), obj_recon.item(), obj_kld.item()))
                    self.pbar.write('D_x:{:.3f} (higher is better)'.format(
                        F.sigmoid(D_x).mean().item()))
                    self.pbar.write('D_x_g:{:.3f} (lower is better)'.format(
                        F.sigmoid(D_x_g).mean().item()))
                    self.pbar.write('running noise sigma:{:.3f}'.format(running_sigma[0]))

                if self.global_iter%self.ckptiter == 0:
                    self.save_checkpoint(str(self.global_iter))
                    if self.dataset == 'dsprites':
                        self.accurate_stat_dict[self.global_iter] = self.calc_mis(running_sigma)

                if self.global_iter%self.ptriter == 0:
                    if self.save_img or self.viz:
                        self.sample_img('fixed')
                        self.sample_img('random')
                        self.traverse()
                        dim_klds = torch.stack(dim_klds)
                        obj_klds = torch.stack(obj_klds).view(-1, 1)
                        obj_recons = torch.stack(obj_recons).view(-1, 1)
                        obj_Gs = torch.stack(obj_Gs).view(-1,1)
                        klds_iters = np.stack(klds_iters)
                        klds = torch.cat([dim_klds, obj_klds], 1).cpu()
                        mis = torch.cat([obj_recons, obj_klds], 1).cpu()
                        Gs = torch.cat([obj_Gs, obj_klds],1).cpu()

                        self.plot_curves(klds_iters, klds, mis,Gs)

                        dim_klds = []
                        obj_klds = []
                        obj_recons = []
                        obj_Gs =[]
                        klds_iters = []

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        stat_path = self.ckpt_dir.joinpath('mis.pth')
        accurate_stat_path = self.ckpt_dir.joinpath('accurate_mis.pth')
        torch.save(self.stat_dict, stat_path.open('wb+'))
        torch.save(self.accurate_stat_dict, accurate_stat_path.open('wb+'))
        self.pbar.write("[*] Training Finished!")
        self.pbar.close()

    def plot_curves(self, X, Y, Z, G):


        if self.win is None:
            self.win = self.viz.line(X=X,
                                     Y=Y,
                                     env=self.viz_name+'_metric',
                                     opts=dict(
                                         xlabel='iteration',
                                         ylabel='kld',))
        else:
            self.win = self.viz.line(X=X,
                                     Y=Y,
                                     env=self.viz_name+'_metric',
                                     win=self.win,
                                     update='append',
                                     opts=dict(
                                         xlabel='iteration',
                                         ylabel='kld',))

        if self.main_win is None:
            self.main_win = self.viz.line(X=X,
                                     Y=Y,
                                     env='a',
                                     opts=dict(
                                         xlabel='iteration',
                                         ylabel='kld',
                                         title=self.viz_name))
        else:
            self.main_win = self.viz.line(X=X,
                                     Y=Y,
                                     env='a',
                                     win=self.main_win,
                                     update='append',
                                     opts=dict(
                                         xlabel='iteration',
                                         ylabel='kld',
                                         title=self.viz_name))

        if self.mi_win is None:
            self.mi_win = self.viz.line(X=X,
                                     Y=Z,
                                     env=self.viz_name+'_metric',
                                     opts=dict(
                                         legend=['lower', 'upper'],
                                         xlabel='iteration',
                                         ylabel='MIs',))
        else:
            self.mi_win = self.viz.line(X=X,
                                     Y=Z,
                                     env=self.viz_name+'_metric',
                                     win=self.mi_win,
                                     update='append',
                                     opts=dict(
                                         legend=['lower', 'upper'],
                                         xlabel='iteration',
                                         ylabel='MIs',))

        if self.main_mi_win is None:
            self.main_mi_win = self.viz.line(X=X,
                                     Y=Z,
                                     env='a',
                                     opts=dict(
                                         legend=['lower', 'upper'],
                                         xlabel='iteration',
                                         ylabel='MIs',
                                         title=self.viz_name))
        else:
            self.main_mi_win = self.viz.line(X=X,
                                     Y=Z,
                                     env='a',
                                     win=self.main_mi_win,
                                     update='append',
                                     opts=dict(
                                         legend=['lower', 'upper'],
                                         xlabel='iteration',
                                         ylabel='MIs',
                                         title=self.viz_name))

        if self.g_win is None:
            self.g_win = self.viz.line(X=X,
                                     Y=G,
                                     env=self.viz_name+'_metric',
                                     opts=dict(
                                         legend=['obj_G', 'upper'],
                                         xlabel='iteration',
                                         ylabel='MIs',))
        else:
            self.g_win = self.viz.line(X=X,
                                     Y=G,
                                     env=self.viz_name+'_metric',
                                     win=self.g_win,
                                     update='append',
                                     opts=dict(
                                         legend=['obj_G', 'upper'],
                                         xlabel='iteration',
                                         ylabel='MIs',))

        if self.main_g_win is None:
            self.main_g_win = self.viz.line(X=X,
                                     Y=G,
                                     env='a',
                                     opts=dict(
                                         legend=['obj_G', 'upper'],
                                         xlabel='iteration',
                                         ylabel='MIs',
                                         title=self.viz_name))
        else:
            self.main_g_win = self.viz.line(X=X,
                                     Y=G,
                                     env='a',
                                     win=self.main_g_win,
                                     update='append',
                                     opts=dict(
                                         legend=['obj_G', 'upper'],
                                         xlabel='iteration',
                                         ylabel='MIs',
                                         title=self.viz_name))



    def traverse(self, edge=1, term=0.1, loc=-1):
        global win_traverse
        self.set_mode('eval')
        interpolation = torch.arange(-edge, edge+0.01, term)
        random_r = Variable(cuda(self.sample_z(1, self.r_dim), self.use_cuda))
        fixed_r = self.fixed_r[0:1]
        R = {'fixed_r':fixed_r, 'random_r':random_r}
        for key in R.keys():
            r_ori = R[key]
            samples = []
            for row in range(self.r_dim):
                if loc!=-1 and row != loc:
                    continue
                r = r_ori.clone()
                for val in interpolation:
                    r[:, row] = val
                    sample = self.unscale(self.G(r0=r))
                    samples.append(sample)
            samples = torch.cat(samples, dim=0).data.cpu()

            title = 'eval_{}_traverse:{}'.format(key, self.global_iter)
            self.viz.images(samples.numpy(), env=self.viz_name+'_eval',
                            opts={'title':title}, nrow=len(interpolation))
            win_traverse = self.viz.images(samples.numpy(), env='evals',
                    opts={'title':'{}:{}_traverse:{}'.format(self.viz_name, key, self.global_iter)}, nrow=len(interpolation), win=win_traverse)

        self.set_mode('train')

    def set_mode(self, mode='train'):
        for net in self.nets:
            if mode == 'train':
                net.train()
            elif mode == 'eval':
                net.eval()
            else:
                raise ValueError("mode should be either 'train' or 'eval'")

    def unscale(self, tensor):
        return tensor.mul(0.5).add(0.5)

    def sample_z(self, batch_size=0, dim=0, dist='normal'):
        if batch_size == 0:
            batch_size = self.batch_size
        if dim == 0:
            dim = self.z_dim

        if dist == 'normal':
            return torch.randn(batch_size, dim)
        elif dist == 'uniform':
            return torch.rand(batch_size, dim).mul(2).add(-1)
        else:
            return None

    def sample_img(self, _type='fixed', nrow=10):
        global win_samples
        self.set_mode('eval')

        if _type == 'fixed':
            r = self.fixed_r
        elif _type == 'random':
            r = self.sample_z(self.sample_num, dim=self.r_dim)
            r = Variable(cuda(r, self.use_cuda))
        else:
            raise ValueError("_type should be either 'fixed' or 'random'")

        samples = self.unscale(self.G(r0=r)).data.cpu()
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
        filename = 'eval_'+_type+'_r:'+str(self.global_iter)
        filepath = self.output_dir.joinpath(filename)

        if self.save_img:
            save_image(grid, str(filepath)+'.jpg')
        if self.viz:
            self.viz.image(grid, env=self.viz_name+'_eval',
                           opts=dict(title=filename, nrow=nrow))
            win_samples = self.viz.image(grid, env='evals',
                           opts=dict(title='{}:{}_samples:{}'.format(self.viz_name, _type, self.global_iter), nrow=nrow), win=win_samples)


        self.set_mode('train')

    def save_checkpoint(self, filename='ckpt.tar'):
        model_states = {'nc':self.nc,
                        'ngf':self.ngf,
                        'ndf':self.ndf,
                        'G':self.G.state_dict(),
                        'DR':self.DR.state_dict()}
        optim_states = {'GR_optim':self.GR_optim.state_dict(),
                        'D_optim':self.D_optim.state_dict()}
        states = {'iter':self.global_iter,
                  'fixed_r':self.fixed_r.cpu(),
                  'fixed_z':self.fixed_z.cpu(),
                  'z_dim':self.z_dim,
                  'r_dim':self.r_dim,
                  'win':self.win,
                  'mi_win':self.mi_win,
                  'g_win':self.g_win,
                  'stat_dict':self.stat_dict,
                  'accurate_stat_dict':self.accurate_stat_dict,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states, file_path.open('wb+'))
        self.pbar.write("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='ckpt.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            checkpoint = torch.load(file_path.open('rb'))
            self.global_iter = checkpoint['iter']
            self.win = checkpoint['win']
            self.mi_win = checkpoint['mi_win']
            self.g_win = checkpoint['g_win']
            self.stat_dict = checkpoint['stat_dict']
            self.accurate_stat_dict = checkpoint['accurate_stat_dict']
            self.fixed_r = cuda(checkpoint['fixed_r'], self.use_cuda)
            self.fixed_z = cuda(checkpoint['fixed_z'], self.use_cuda)
            self.DR.load_state_dict(checkpoint['model_states']['DR'])
            self.G.load_state_dict(checkpoint['model_states']['G'])
            self.GR_optim.load_state_dict(checkpoint['optim_states']['GR_optim'])
            self.D_optim.load_state_dict(checkpoint['optim_states']['D_optim'])
            self.pbar.write(self.global_iter)
            self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            self.pbar.write("=> no checkpoint found at '{}'".format(file_path))

    def project_r(self, img, iters=100):
        totensor = transforms.ToTensor()
        pil_img = Image.open('data/3DChairs/657/image_057_p030_t290_r096.png')
        img = totensor().unsqueeze(0) # 1, 3, 64, 64
        r = torch.Tensor(1, self.r_dim)
        r.normal_(0,0.1)
        r.requires_grad = True
        adam = optim.Adam([r])
        self.G.eval()
        for i in iters:
            self.GR_optim.zero_grad()
            adam.zero_grad()
            img_recon = self.GR(None, r)
            loss = F.mse_loss(img_recon, img)
            loss.backward()
            adam.step()
        save_image('projected.jpg', img_recon[0])
        return r


