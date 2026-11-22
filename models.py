"""DCGAN generator and spectral-norm discriminator for square images of side 2^k (k >= 5)."""
import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class Generator(nn.Module):
    def __init__(self, nz=128, ngf=64, res=64):
        super().__init__()
        n_up = int(math.log2(res)) - 3  # 4x4 -> res/2 via stride-2 blocks, then final block to res
        c = ngf * 2 ** n_up
        layers = [nn.ConvTranspose2d(nz, c, 4, 1, 0, bias=False), nn.BatchNorm2d(c), nn.ReLU(True)]
        for _ in range(n_up):
            layers += [nn.ConvTranspose2d(c, c // 2, 4, 2, 1, bias=False), nn.BatchNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        layers += [nn.ConvTranspose2d(c, 3, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.nz = nz

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, ndf=64, res=64):
        super().__init__()
        n_down = int(math.log2(res)) - 3
        c = ndf
        layers = [spectral_norm(nn.Conv2d(3, c, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        for _ in range(n_down):
            layers += [spectral_norm(nn.Conv2d(c, c * 2, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
            c *= 2
        layers += [spectral_norm(nn.Conv2d(c, 1, 4, 1, 0))]  # 4x4 -> 1x1 logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


def init_weights(m):
    """DCGAN init: N(0, 0.02) convs, N(1, 0.02) BN scales."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        w = m.parametrizations.weight.original if hasattr(m, "parametrizations") else m.weight
        nn.init.normal_(w, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)
