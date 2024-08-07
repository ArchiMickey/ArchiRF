import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp, Attention
import torch.nn.functional as F
from typing import Tuple
from functools import partial
from einops import repeat, pack, unpack, rearrange

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# Positional embedding from:
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class MultiTokenLabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim, dropout_prob, num_embeddings=2):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embeddings = nn.ModuleList()
        for _ in range(num_embeddings):
            self.embeddings.append(
                nn.Embedding(num_classes + int(use_cfg_embedding), dim)
            )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def token_drop(self, labels, force_drop_ids=None):
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
        embeddings = [emb(labels) for emb in self.embeddings]
        embeddings, _ = pack(embeddings, "b * d")
        global_embeddings = self.mlp(embeddings.mean(1))
        return embeddings, global_embeddings


class IN1KEmbedder(nn.Module):
    def __init__(self, dim, embed_path, dropout_prob):
        super().__init__()
        self.num_classes = 1000
        self.dropout_prob = dropout_prob

        self.embed = nn.Parameter(torch.load(embed_path))

        self.proj = nn.Linear(1280, dim)
        self.mlp = nn.Sequential(nn.Linear(1280, dim), nn.SiLU(), nn.Linear(dim, dim))

    def token_drop(self, labels, force_drop_ids=None):
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
        null_idx = torch.where(labels == self.num_classes)
        labels[null_idx] = 0  # replace null token with 0 to avoid out of index error
        _embeddings = self.embed[labels]
        _embeddings[null_idx] = _embeddings[null_idx] * 0
        embeddings = self.proj(_embeddings)
        global_embeddings = self.mlp(_embeddings.mean(1))
        return embeddings, global_embeddings


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        dtype = x.dtype
        x = F.normalize(x, dim=-1) * self.scale * self.g
        return x.to(dtype)


class DoubleAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm_1 = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_1 = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.qkv_2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm_2 = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_2 = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, c):
        B, N1, C = x.shape
        B, N2, C = c.shape
        qkv1 = self.qkv_1(x).chunk(3, dim=-1)
        q1, k1, v1 = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv1
        )
        q1, k1 = self.q_norm_1(q1), self.k_norm_1(k1)

        qkv2 = self.qkv_2(c).chunk(3, dim=-1)
        q2, k2, v2 = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv2
        )
        q2, k2 = self.q_norm_2(q2), self.k_norm_2(k2)

        q, k, v = (
            torch.cat([q1, q2], dim=-2),
            torch.cat([k1, k2], dim=-2),
            torch.cat([v1, v2], dim=-2),
        )

        if (
            x.dtype in [torch.float16, torch.bfloat16]
            and flash_attn_func is not None
        ):
            q, k, v = map(lambda t: rearrange(t, "b h n d -> b n h d"), (q, k, v))
            x = flash_attn_func(q, k, v, self.attn_drop.p if self.training else 0)
            x = rearrange(x, "b n h d -> b n (h d)")
        else:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0
            )
            x = rearrange(x, "b h n d -> b n (h d)")

        x, c = x.split([N1, N2], dim=1)
        x = self.proj_1(x)
        c = self.proj_2(c)

        return x, c


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x

class MMDiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1_1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm1_2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = DoubleAttention(
            dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm
        )
        self.norm2_1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp_1 = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.mlp_2 = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )

        self.adaLN_modulation_1 = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.adaLN_modulation_2 = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c, global_c):
        (shift_msa_1, scale_msa_1, gate_msa_1, shift_mlp_1, scale_mlp_1, gate_mlp_1) = (
            self.adaLN_modulation_1(global_c).chunk(6, dim=-1)
        )
        (shift_msa_2, scale_msa_2, gate_msa_2, shift_mlp_2, scale_mlp_2, gate_mlp_2) = (
            self.adaLN_modulation_2(global_c).chunk(6, dim=-1)
        )
        attn_x, attn_c = self.attn(
            modulate(self.norm1_1(x), scale_msa_1, shift_msa_1),
            modulate(self.norm1_2(c), scale_msa_2, shift_msa_2),
        )
        x = x + gate_msa_1.unsqueeze(1) * attn_x
        c = c + gate_msa_2.unsqueeze(1) * attn_c

        x = x + gate_mlp_1.unsqueeze(1) * self.mlp_1(
            modulate(self.norm2_1(x), scale_mlp_1, shift_mlp_1)
        )
        c = c + gate_mlp_2.unsqueeze(1) * self.mlp_2(
            modulate(self.norm2_2(c), scale_mlp_2, shift_mlp_2)
        )
        return x, c


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MMDiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=1152,
        depth=14,
        depth_single_blocks=14,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=0,
        class_dropout_prob=0.1,
        num_classes=1000,
        num_embeddings=2,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.use_cond = True

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)

        self.y_embedder = MultiTokenLabelEmbedder(
            num_classes, dim, class_dropout_prob, num_embeddings
        )

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_embeddings + num_patches, dim), requires_grad=False
        )

        # register tokens
        self.register_tokens = (
            nn.Parameter(torch.randn(num_register_tokens, dim))
            if num_register_tokens > 0
            else None
        )

        self.double_blocks = nn.ModuleList(
            [MMDiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth_single_blocks)]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 7)
        self.pos_embed[:, : self.num_embeddings].data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed[:, self.num_embeddings :].data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        for emb in self.y_embedder.embeddings:
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.double_blocks:
            nn.init.constant_(block.adaLN_modulation_1[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_1[-1].bias, 0)
            nn.init.constant_(block.adaLN_modulation_2[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_2[-1].bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize register tokens:
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def prepare_tokens(self, x, y):
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        x = x + self.pos_embed[:, self.num_embeddings :]

        y, global_y = self.y_embedder(y, self.training)  # (N, D)
        y = y + self.pos_embed[:, : self.num_embeddings]

        if self.register_tokens is not None:
            # repeat register token
            r = repeat(self.register_tokens, "n d -> b n d", b=x.shape[0])

            # pack cls token and register token
            x = torch.cat([r, x], dim=1)

        return x, y, global_y

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        use_register_tokens = self.register_tokens is not None

        x, y, global_y = self.prepare_tokens(x, y)

        t = self.t_embedder(t)  # (N, D)
        global_c = t
        global_c = global_c + global_y  # (N, D)

        for block in self.double_blocks:
            x, y = block(x, y, global_c)  # (N, T, D)
        
        x = torch.cat([y, x], dim=1)
        
        for block in self.blocks:
            x = block(x, global_c)
        
        x = x[:, y.shape[1] :]
        
        # unpack cls token and register token
        if use_register_tokens:
            x = x[:, self.num_register_tokens :]

        x = self.final_layer(x, global_c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        x = x.repeat(2, 1, 1, 1)
        t = t.repeat(2)
        y = y.repeat(2)
        y[len(x) // 2 :] = self.y_embedder.num_classes

        model_out = self.forward(x, t, y)
        cond_eps, uncond_eps = model_out.split(len(x) // 2)
        out = uncond_eps + (cond_eps - uncond_eps) * cfg_scale
        return out


class IN1KMMDiT(MMDiT):
    def __init__(
        self,
        embed_path,
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=1152,
        depth=14,
        depth_single_blocks=14,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=0,
        class_dropout_prob=0.1,
        num_classes=1000,
        num_embeddings=49,
        learn_sigma=False,
    ):
        super().__init__(
            input_size,
            patch_size,
            in_channels,
            dim,
            depth,
            depth_single_blocks,
            num_heads,
            mlp_ratio,
            num_register_tokens,
            class_dropout_prob,
            num_classes,
            num_embeddings,
            learn_sigma,
        )
        self.y_embedder = IN1KEmbedder(dim, embed_path, class_dropout_prob)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.proj.weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)
