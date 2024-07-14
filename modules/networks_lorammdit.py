import torch
import loralib as lora
from copy import deepcopy
from einops import rearrange, pack
from modules.dit_original.models import *

import torch.nn as nn
import torch.nn.functional as F


def inject_lora(module, lora_config):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            lora_linear = lora.Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                **lora_config,
            )
            lora_linear.load_state_dict(child.state_dict(), strict=False)
            module.__setattr__(name, lora_linear)
        elif isinstance(child, nn.Conv2d):
            lora_conv2d = lora.Conv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size[0],
                stride=child.stride,
                padding=child.padding,
                **lora_config,
            )
            lora_conv2d.conv.load_state_dict(child.state_dict(), strict=False)
            module.__setattr__(name, lora_conv2d)
        elif isinstance(child, nn.Embedding):
            lora_embedding = lora.Embedding(
                num_embeddings=child.num_embeddings,
                embedding_dim=child.embedding_dim,
                **lora_config,
            )
            lora_embedding.load_state_dict(child.state_dict(), strict=False)
            module.__setattr__(name, lora_embedding)
        else:
            inject_lora(child, lora_config)


def inject_lora_to_module(module, lora_config):
    lora_module = deepcopy(module)
    inject_lora(lora_module, lora_config)
    return lora_module


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


class DoubleAttention_from_Attention(nn.Module):
    def __init__(self, attn):
        super().__init__()
        dim = 1152
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scale = attn.scale
        self.fused_attn = attn.fused_attn

        self.qkv_x = attn.qkv
        self.q_norm_x = RMSNorm(dim)
        self.k_norm_x = RMSNorm(dim)
        self.proj_x = attn.proj

        self.qkv_c = nn.Linear(dim, dim * 3, bias=self.qkv_x.bias is not None)
        self.q_norm_c = RMSNorm(dim)
        self.k_norm_c = RMSNorm(dim)
        self.proj_c = nn.Linear(dim, dim, bias=self.proj_x.bias is not None)

        self.attn_drop = attn.attn_drop
        self.proj_drop_x, self.proj_drop_c = attn.proj_drop, attn.proj_drop

    def forward(self, x, c):
        B, N1, C = x.shape
        qkv_x = self.qkv_x(x).chunk(3, dim=-1)
        q_x, k_x, v_x = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv_x
        )
        q_x, k_x = self.q_norm_x(q_x), self.k_norm_x(k_x)

        B, N2, C = c.shape
        qkv_c = self.qkv_c(c).chunk(3, dim=-1)
        q_c, k_c, v_c = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv_c
        )
        q_c, k_c = self.q_norm_c(q_c), self.k_norm_c(k_c)

        q, k, v = (
            torch.cat((q_x, q_c), dim=-2),
            torch.cat((k_x, k_c), dim=-2),
            torch.cat((v_x, v_c), dim=-2),
        )

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = rearrange(x, "b h n d -> b n (h d)")
        x, c = x.split([N1, N2], dim=1)
        x = self.proj_x(x)
        x = self.proj_drop_x(x)
        c = self.proj_c(c)
        c = self.proj_drop_c(c)

        return x, c


class MMDiTBlock_from_DiTBlock(nn.Module):
    def __init__(self, block):
        super().__init__()        
        for name, child in block.named_children():
            if name != "attn":
                self.__setattr__(f"{name}_x", child)
        hidden_size = 1152
        mlp_ratio = 4.0
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.norm1_c = nn.LayerNorm(hidden_size)
        self.norm2_c = nn.LayerNorm(hidden_size)
        self.mlp_c = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation_c = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.attn = DoubleAttention_from_Attention(block.attn)

    def forward(self, x, c, global_c):
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = (
            self.adaLN_modulation_x(global_c).chunk(6, dim=1)
        )
        shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = (
            self.adaLN_modulation_c(global_c).chunk(6, dim=1)
        )

        attn_x, attn_c = self.attn(
            modulate(self.norm1_x(x), shift_msa_x, scale_msa_x),
            modulate(self.norm1_c(c), shift_msa_c, scale_msa_c),
        )

        x = x + gate_msa_x.unsqueeze(1) * attn_x
        c = c + gate_msa_c.unsqueeze(1) * attn_c

        x = x + gate_mlp_x.unsqueeze(1) * self.mlp_x(
            modulate(self.norm2_x(x), shift_mlp_x, scale_mlp_x)
        )
        c = c + gate_mlp_c.unsqueeze(1) * self.mlp_c(
            modulate(self.norm2_c(c), shift_mlp_c, scale_mlp_c)
        )
        return x, c

class MultiTokenLabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding1 = nn.Embedding(num_classes + use_cfg_embedding, dim)
        self.embedding2 = nn.Embedding(num_classes + use_cfg_embedding, dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

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
        emb1 = self.embedding1(labels)
        emb2 = self.embedding2(labels)
        embeddings, _ = pack([emb1, emb2], "b * d")
        global_embeddings = self.mlp(torch.cat([emb1, emb2], dim=-1))
        return embeddings, global_embeddings

class LoRAMMDiT(nn.Module):
    def __init__(self, num_classes, dit_ckpt_path="/mnt/g/DiT-XL-2-256x256.pt", lora_config={}):
        super().__init__()
        net = DiT_models["DiT-XL/2"]()
        ckpt = torch.load(dit_ckpt_path, map_location="cpu")
        net.load_state_dict(ckpt)
        net.requires_grad_(False)

        self.num_classes = num_classes
        self.out_channels = 4

        self.x_embedder = inject_lora_to_module(net.x_embedder, lora_config)
        self.t_embedder = TimestepEmbedder(1152)
        self.y_embedder = MultiTokenLabelEmbedder(num_classes, 1152, 0.1)

        self.pos_embed = net.pos_embed

        self.blocks = inject_lora_to_module(net.blocks, lora_config)
        self.blocks = nn.ModuleList(
            [MMDiTBlock_from_DiTBlock(block) for block in self.blocks]
        )
        self.final_layer = FinalLayer(1152, 2, 4)

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

    def forward(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        global_c = t
        y, global_y = self.y_embedder(y, self.training)
        global_c = global_c + global_y

        for i, block in enumerate(self.blocks):
            x, y = block(x, y, global_c)

        x = self.final_layer(x, global_c)
        x = self.unpatchify(x)
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