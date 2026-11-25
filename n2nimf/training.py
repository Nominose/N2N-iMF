"""High-pass PatchGAN used to fine-tune the brain CT and PCCT generators."""

import torch
import torch.nn.functional as F
from torch import nn


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b, stride in [(1, 64, 2), (64, 128, 2), (128, 256, 2), (256, 512, 1)]:
            layers.extend(
                [nn.Conv2d(a, b, 4, stride, 1), nn.LeakyReLU(0.2, inplace=True)]
            )
        self.net = nn.Sequential(*layers, nn.Conv2d(512, 1, 4, 1, 1))

    def forward(self, x):
        residual = x - F.avg_pool2d(x, 7, stride=1, padding=3)
        return self.net(residual)


def discriminator_loss(discriminator, real, fake, step, gamma):
    regularize = step % 16 == 0
    real = real.detach().requires_grad_(regularize)
    positive = discriminator(real)
    negative = discriminator(fake.detach())
    loss = (1 - positive).relu().mean() + (1 + negative).relu().mean()
    if regularize:
        (gradient,) = torch.autograd.grad(positive.sum(), real, create_graph=True)
        loss = loss + 8 * gamma * gradient.flatten(1).square().sum(1).mean()
    return loss
