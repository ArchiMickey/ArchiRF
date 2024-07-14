import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class RMSNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = RMSNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = RMSNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return self.skip(x) + h

class Downsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = x.shape[-2:]
        return F.interpolate(x, (h // 2, w // 2), mode="bilinear")

class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = x.shape[-2:]
        return F.interpolate(x, (h * 2, w * 2), mode="bilinear")

class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.num_heads = max(1, channels // 64)
        self.head_dim = channels // self.num_heads
        self.scale = channels ** -0.5
        
        self.norm = RMSNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        h, w = x.shape[-2:]
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.num_heads), (q, k, v))
        q, k = self.q_norm(q), self.k_norm(k)
        
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = rearrange(out, "b h (x y) c -> b (h c) x y", x=h, y=w)
        out = self.proj(out)
        
        return x + out

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class Decoder(nn.Module):
    def __init__(self, channels, channel_mult, n_res_blocks, out_channels, z_channels, **kwargs):
        super().__init__()
        num_resolutions = len(channel_mult)
        channel_list = [m * channels for m in channel_mult]
        reversed(channel_list)
        channels = channel_list[0]
        
        self.conv_in = nn.Conv2d(z_channels, channels, 3, padding=1)
        
        self.mid_block_1 = ResnetBlock(channels, channels)
        self.mid_attn = AttnBlock(channels)
        self.mid_block_2 = ResnetBlock(channels, channels)
        
        self.ups = nn.ModuleList()
        for i in range(num_resolutions):
            blocks = nn.ModuleList()
            for _ in range(n_res_blocks+1):
                blocks.append(ResnetBlock(channels, channel_list[i]))
                channels = channel_list[i]
            upsample = Upsample() if i != num_resolutions - 1 else nn.Identity()
            
            up = nn.Module()
            up.blocks = blocks
            up.upsample = upsample
            self.ups.append(up)
        self.norm_out = RMSNorm2d(channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, padding=1)
    
    def forward(self, z):
        h = self.conv_in(z)
        
        h = self.mid_block_1(h)
        h = self.mid_attn(h)
        h = self.mid_block_2(h)
        
        for up in self.ups:
            for block in up.blocks:
                h = block(h)
            h = up.upsample(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

class Encoder(nn.Module):
    def __init__(self, channels, channel_mult, n_res_blocks, in_channels, z_channels, **kwargs):
        super().__init__()
        n_resolutions = len(channel_mult)
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        channel_list = [m * channels for m in channel_mult]
        channels = channel_list[0]
        
        self.downs = nn.ModuleList()
        for i in range(n_resolutions):
            blocks = nn.ModuleList()
            for _ in range(n_res_blocks):
                blocks.append(ResnetBlock(channels, channel_list[i]))
                channels = channel_list[i]
            downsample = Downsample() if i != n_resolutions - 1 else nn.Identity()
            
            down = nn.Module()
            down.blocks = blocks
            down.downsample = downsample
            self.downs.append(down)
        
        self.mid_block_1 = ResnetBlock(channels, channels)
        self.mid_attn = AttnBlock(channels)
        self.mid_block_2 = ResnetBlock(channels, channels)
        
        self.norm_out = RMSNorm2d(channels)
        self.conv_out = nn.Conv2d(channels, z_channels, 1)
    
    def forward(self, x):
        x = self.conv_in(x)
        
        for down in self.downs:
            for block in down.blocks:
                x = block(x)
            x = down.downsample(x)
        
        x = self.mid_block_1(x)
        x = self.mid_attn(x)
        x = self.mid_block_2(x)
        
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return x
