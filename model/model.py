"""model.py"""

import torch
import torch.nn as nn
from torch.autograd import Variable

def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    vector = mu + std*eps

    return vector


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Generator(nn.Module):
    def __init__(self, z_dim=100, r_dim=10, nc=1, ngf=64, metric=None):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.r_dim = r_dim
        self.metric = metric
        emb_fn = 64
        #emb_fn = 500
        self.emb = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, r_dim*2),
        )
        self.fe = nn.Sequential(
            nn.Linear(r_dim, ngf*16),
            nn.BatchNorm1d(ngf*16),
            nn.ReLU(True),
            nn.Linear(ngf*16, 8*8*ngf*4),
            nn.BatchNorm1d(8*8*ngf*4),
            nn.ReLU(True),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*4, 3, 1, 1, bias=False), # originally 4x4 conv_trans
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*4, 3, 1, 1, bias=False), # originally 4x4 conv_trans
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z=None, r0=None):
        # when traverse
        if r0 is not None:
            out = self.fe(r0)
            out = out.view(out.size(0), -1, 8, 8)
            out = self.conv(out)
            return out

        stat = self.emb(z)
        r_mu = stat[:, :self.r_dim]
        r_logvar = stat[:, self.r_dim:]

        # when eval
        if self.metric == 'factorvae':
            return r_mu
        elif self.metric == 'betatcvae':
            return torch.cat([r_mu, r_logvar.div(2)], 1)

        # when training
        else:
            r = reparametrize(r_mu, r_logvar)
            out = self.fe(r)
            out = out.view(out.size(0), -1, 8, 8)
            out = self.conv(out)

            return r_mu, r_logvar, r, out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            #nn.init.xavier_normal_(m)

class Discriminator(nn.Module):
    def __init__(self, z_dim=100, nc=1, ndf=64, metric=None):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.metric = metric
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 3, 1, 1, bias=False), # originally, 4x4 conv
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, ndf*16, 8,  bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*16, ndf*16, 1),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*16, z_dim, 1),
        )
        self.disc = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 3, 1, 1, bias=False), # originally, 4x4 conv
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, ndf*16, 8,  bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*16, 1, 1)
        )

    def forward(self, x, z=None):

        disc_out = self.disc(x).view(x.size(0), 1)
        enc_out = self.encoder(x).view(x.size(0), self.z_dim)

        if self.metric == 'factorvae':
            # when eval factorvae metric
            return enc_out
        elif self.metric == 'betatcvae':
            # when eval betatcvae metric
            return enc_out
        else:
            # when training
            return enc_out, disc_out


    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        #if m.bias.data is not None:
        #    m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        #if m.bias.data is not None:
        #    m.bias.data.zero_()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        print(classname)
        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        print(classname)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        print(classname)
        if m.bias is not None:
            m.bias.data.zero_()

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight)
        print(classname)
        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        print(classname)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight)
        print(classname)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    batch_size = 5
    z_dim = 100

    G = Generator(z_dim)
    R = Reconstructor(z_dim)
    D = Discriminator()
    import ipdb; ipdb.set_trace()
