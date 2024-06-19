import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange
from typing import List, Optional


def Upsample():
    return nn.Upsample(scale_factor=2, mode="bilinear")


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.interpolate(x, size=(h // 2, w // 2), mode="bilinear")
        x = self.conv(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = RMSNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (1 + scale) + shift

        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim=None, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.in_block = Block(in_channels, out_channels)

        self.emb_layers = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))
            if time_emb_dim
            else None
        )

        self.out_block = Block(out_channels, out_channels)

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, 1)
            if out_channels != in_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        scale_shift = None
        if time_emb is not None:
            cond_emb = self.emb_layers(time_emb)
            cond_emb = rearrange(cond_emb, "b c -> b c 1 1")
            scale_shift = cond_emb.chunk(2, dim=1)
        h = self.in_block(x, scale_shift)
        h = self.out_block(h)
        return h + self.skip_connection(x)


class RMSNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.norm = RMSNorm2d(dim)

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.q_norm = RMSNorm2d(dim)
        self.k_norm = RMSNorm2d(dim)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), (q, k, v)
        )

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h (x y) c -> b (h c) x y", x=h, y=w)
        return self.to_out(out)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, t_emb):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        return x


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        channel_multipliers: List[int],
        attention_levels: List[int] = [],
        n_heads: int = 1,
        pos_emb_theta: int = 10000,
        n_res_block: int = 1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        levels = len(channel_multipliers)

        emb_dim = channels * 4
        self.t_embedder = TimestepEmbedder(emb_dim, pos_emb_theta)
        self.y_embedder = (
            LabelEmbedder(num_classes, emb_dim, dropout_prob=0.1)
            if num_classes
            else None
        )

        self.init_conv = nn.Conv2d(in_channels, channels, 3, padding=1)
        input_block_channels = [channels]
        channel_list = [channels * m for m in channel_multipliers]
        self.downs = nn.ModuleList()
        for i in range(levels):
            for _ in range(n_res_block):
                layers = [ResBlock(channels, emb_dim, out_channels=channel_list[i])]
                channels = channel_list[i]
                if i in attention_levels:
                    layers.append(Attention(channels, n_heads))
                input_block_channels.append(channels)
                self.downs.append(TimestepEmbedSequential(*layers))

            if i != levels - 1:
                self.downs.append(TimestepEmbedSequential(Downsample(channels)))
                input_block_channels.append(channels)

        self.mid_block = TimestepEmbedSequential(
            ResBlock(channels, emb_dim),
            Attention(channels, n_heads),
            ResBlock(channels, emb_dim),
        )

        self.ups = nn.ModuleList()
        for i in reversed(range(levels)):
            for j in range(n_res_block + 1):
                layers = [
                    ResBlock(
                        channels + input_block_channels.pop(),
                        emb_dim,
                        out_channels=channel_list[i],
                    )
                ]
                channels = channel_list[i]

                if i in attention_levels:
                    layers.append(Attention(channels, n_heads))

                if i != 0 and j == n_res_block:
                    layers.append(Upsample())
                self.ups.append(TimestepEmbedSequential(*layers))

        self.out = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, x, t, y=None):
        hs = []

        c = self.t_embedder(t)
        if y is not None:
            if not self.y_embedder:
                raise ValueError("Model not configured to handle class labels.")
            c = c + self.y_embedder(y, self.training)

        x = self.init_conv(x)
        hs.append(x)

        for module in self.downs:
            x = module(x, c)
            hs.append(x)

        x = self.mid_block(x, c)

        for module in self.ups:
            x = torch.cat([x, hs.pop()], dim=1)
            x = module(x, c)

        return self.out(x)

    def forward_with_cfg(self, x, t, y, cfg_scale=1.0):
        x = x.repeat(2, 1, 1, 1)
        t = t.repeat(2)
        y = y.repeat(2)
        y[len(x) // 2 :] = self.y_embedder.num_classes

        model_out = self.forward(x, t, y)
        cond_eps, uncond_eps = model_out.split(len(x) // 2)
        eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return eps
