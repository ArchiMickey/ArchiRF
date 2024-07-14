from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from modules.pg_modules.blocks import DownBlock, DownBlockPatch, conv2d
from modules.pg_modules.projector import F_RandomProj
from modules.pg_modules.diffaug import DiffAugment
from modules.utils import ActNorm


class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False):
        super().__init__()
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        return self.main(x)


class SingleDiscCond(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False, c_dim=1000, cmap_dim=64, embedding_dim=128):
        super().__init__()
        self.cmap_dim = cmap_dim

        # midas channels
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2
        self.main = nn.Sequential(*layers)

        # additions for conditioning on class information
        self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)
        self.embed = nn.Embedding(num_embeddings=c_dim, embedding_dim=embedding_dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(self.embed.embedding_dim, self.cmap_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, c):
        h = self.main(x)
        out = self.cls(h)

        # conditioning via projection
        cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out


class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        num_discs=1,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        separable=False,
        patch=False,
        **kwargs,
    ):
        super().__init__()

        assert num_discs in [1, 2, 3, 4]

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDiscCond if cond else SingleDisc

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, separable=separable, patch=patch)],
        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c):
        all_logits = []
        for k, disc in self.mini_discs.items():
            all_logits.append(disc(features[k], c).view(features[k].size(0), -1))

        all_logits = torch.cat(all_logits, dim=1)
        return all_logits


class ProjectedDiscriminator(torch.nn.Module):
    def __init__(
        self,
        diffaug=True,
        interp224=True,
        backbone_kwargs={},
        **kwargs
    ):
        super().__init__()
        self.diffaug = diffaug
        self.interp224 = interp224
        self.feature_network = F_RandomProj(**backbone_kwargs)
        self.discriminator = MultiScaleD(
            channels=self.feature_network.CHANNELS,
            resolutions=self.feature_network.RESOLUTIONS,
            **backbone_kwargs,
        )

    def train(self, mode=True):
        self.feature_network = self.feature_network.train(False)
        self.discriminator = self.discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x, c=None):
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')

        if self.interp224:
            x = F.interpolate(x, 224, mode='bilinear', align_corners=False)

        features = self.feature_network(x)
        logits = self.discriminator(features, c)

        return logits

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)
        self.apply(weights_init)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)