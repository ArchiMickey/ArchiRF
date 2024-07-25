import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm.auto import tqdm

# from diffusers.models import AutoencoderKL
from autoencoder.model import AutoencoderKL, Autoencoder

from modules.utils import normalize_to_neg1_1, unnormalize_to_0_1


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        net,
        channels=3,
        image_size=32,
        logit_normal_sampling_t=False,
        normalize_input_output=True
    ):
        super().__init__()
        self.net = net
        self.use_cond = self.net.num_classes is not None
        self.channels = channels
        self.image_size = image_size
        self.logit_normal_sampling_t = logit_normal_sampling_t
        self.normalize_input_output = normalize_input_output

    def forward(self, x, c=None):
        if self.use_cond:
            assert c is not None, "Conditional model requires class labels"

        if self.logit_normal_sampling_t:
            t = torch.randn((x.shape[0],), device=x.device).sigmoid()
        else:
            t = torch.rand((x.shape[0],), device=x.device)

        t_ = rearrange(t, "b -> b 1 1 1")
        z = torch.randn_like(x)
        if self.normalize_input_output:
            x = normalize_to_neg1_1(x)
        z_t = (1 - t_) * x + t_ * z
        v_t = self.net(z_t, t, c)
        target = z - x
        return F.mse_loss(v_t, target)

    @torch.inference_mode()
    def sample(self, batch_size, device, sampling_steps=50, return_all_steps=False):
        z = torch.randn(
            (batch_size, self.channels, self.image_size, self.image_size),
            device=device,
        )

        images = [z]
        t_span = torch.linspace(0, 1, sampling_steps, device=device)
        for t in tqdm(reversed(t_span), leave=False, dynamic_ncols=True):
            v_t = self.net(z, t)
            z = z - v_t / sampling_steps
            images.append(z)

        if self.normalize_input_output:
            z = unnormalize_to_0_1(z.clip(-1, 1))

        if return_all_steps:
            all_steps = torch.stack(images)
            if self.normalize_input_output:
                all_steps = unnormalize_to_0_1(all_steps.clip(-1, 1))
            return z, all_steps
        return z

    @torch.inference_mode()
    def cond_sample(
        self, classes, device, sampling_steps=50, cfg_scale=5.0, return_all_steps=False
    ):
        assert self.use_cond
        y = torch.tensor(classes, device=device)
        z = torch.randn(
            (len(classes), self.channels, self.image_size, self.image_size),
            device=device,
        )

        images = [z]
        t_span = torch.linspace(0, 1, sampling_steps, device=device)
        for t in tqdm(reversed(t_span), leave=False, dynamic_ncols=True):
            t = t.repeat(len(z))
            v_t = self.net.forward_with_cfg(z, t, y, cfg_scale)
            z = z - v_t / sampling_steps
            images.append(z)

        if self.normalize_input_output:
            z = unnormalize_to_0_1(z.clip(-1, 1))

        if return_all_steps:
            all_steps = torch.stack(images)
            if self.normalize_input_output:
                all_steps = unnormalize_to_0_1(all_steps.clip(-1, 1))
            return z, all_steps
        return z

    def fid_sample(self, batch_size, device, cfg_scale=5.0):
        if self.use_cond:
            y = torch.randint(0, self.net.num_classes, (batch_size,), device=device)

            return self.cond_sample(y, device, cfg_scale=cfg_scale)
        else:
            return self.sample(batch_size, device)


class LatentRectifiedFlow(RectifiedFlow):
    def __init__(
        self,
        net,
        channels=3,
        image_size=32,
        logit_normal_sampling_t=False,
    ):
        super().__init__(net, channels, image_size, logit_normal_sampling_t)
        self.vae = Autoencoder.from_pretrained("autoencoder/autoencoder_ema.ckpt")
        for p in self.vae.parameters():
            p.requires_grad = False
        
    @torch.inference_mode()
    def encode(self, x):
        x = normalize_to_neg1_1(x)
        return self.vae.encode(x)
    
    @torch.inference_mode()
    def decode(self, z):
        z = z.clip(-1, 1)
        x_ = self.vae.decode(z)
        return unnormalize_to_0_1(x_.clip(-1, 1))
    
    def forward(self, x, c=None):
        if self.use_cond:
            assert c is not None, "Conditional model requires class labels"

        if self.logit_normal_sampling_t:
            t = torch.randn((x.shape[0],), device=x.device).sigmoid()
        else:
            t = torch.rand((x.shape[0],), device=x.device)

        t_ = rearrange(t, "b -> b 1 1 1")
        x = self.encode(x)
        z = torch.randn_like(x)
        z_t = (1 - t_) * x + t_ * z
        v_t = self.net(z_t, t, c)
        target = z - x
        return F.mse_loss(target, v_t)
    
    @torch.inference_mode()
    def sample(self, batch_size, device, sampling_steps=50, return_all_steps=False):
        z = torch.randn(
            (batch_size, self.channels, self.image_size, self.image_size),
            device=device,
        )

        images = [z]
        t_span = torch.linspace(0, 1, sampling_steps, device=device)
        for t in tqdm(reversed(t_span), leave=False, dynamic_ncols=True):
            v_t = self.net(z, t)
            z = z - v_t / sampling_steps
            images.append(z)

        z = self.decode(z)

        if return_all_steps:
            all_steps = [self.decode(img) for img in images]
            all_steps = torch.stack(all_steps)
            return z, all_steps
        return z

    @torch.inference_mode()
    def cond_sample(
        self, classes, device, sampling_steps=50, cfg_scale=5.0, return_all_steps=False
    ):
        assert self.use_cond
        y = torch.tensor(classes, device=device)
        z = torch.randn(
            (len(classes), self.channels, self.image_size, self.image_size),
            device=device,
        )

        images = [z]
        t_span = torch.linspace(0, 1, sampling_steps, device=device)
        for t in tqdm(reversed(t_span), leave=False, dynamic_ncols=True):
            t = t.repeat(len(z))
            v_t = self.net.forward_with_cfg(z, t, y, cfg_scale)
            z = z - v_t / sampling_steps
            images.append(z)

        z = self.decode(z)

        if return_all_steps:
            all_steps = [self.decode(img) for img in images]
            all_steps = torch.stack(all_steps)
            return z, all_steps
        return z