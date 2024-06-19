import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

from tqdm.auto import tqdm, trange

from modules.utils import normalize_to_neg1_1, unnormalize_to_0_1


class Diffusion(nn.Module):
    def __init__(self, net, betas, n_T, channels=3, image_size=32, ddim_eta=0.0):
        super(Diffusion, self).__init__()
        self.net = net
        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
        # for k, v in flow_to_diffusion(0.01, n_T).items():
        #    self.register_buffer(k, v)

        self.n_T = n_T
        self.channels = channels
        self.image_size = image_size
        self.use_cond = self.net.num_classes is not None
        self.ddim_eta = ddim_eta

    def forward(self, x, c=None):
        """
        this method is used in training, so samples t and noise randomly
        """
        if self.use_cond:
            assert c is not None, "Conditional model requires class labels"

        # choosing time step
        t = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)

        x = normalize_to_neg1_1(x)  # normalize to [-1, 1]

        # noise
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[t, None, None, None] * x
            + self.sqrtmab[t, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return F.mse_loss(eps, self.net(x_t, t, c))

    @torch.inference_mode()
    def sample(self, batch_size, device, sampling_steps=50, return_all_steps=False):

        if sampling_steps < self.n_T:
            return self.ddim_sample(
                batch_size, device, sampling_steps, return_all_steps
            )

        x_i = torch.randn(
            (batch_size, self.channels, self.image_size, self.image_size),
            device=device,
        )

        imgs = [x_i]

        for i in trange(
            self.n_T,
            0,
            -1,
            desc=f"Sampling...",
            leave=False,
            dynamic_ncols=True,
        ):
            z = torch.randn_like(x_i) if i > 1 else 0

            t = torch.tensor([i] * len(x_i)).to(device)
            eps = self.net(x_i, t)

            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

            imgs.append(x_i)

        x_i = unnormalize_to_0_1(x_i.clip(-1, 1))

        if return_all_steps:
            return x_i, unnormalize_to_0_1(torch.stack(imgs).clip(-1, 1))

        return x_i

    @torch.inference_mode()
    def ddim_sample(
        self, batch_size, device, sampling_steps=50, return_all_steps=False
    ):
        timesteps = torch.tensor(list(range(0, self.n_T, self.n_T // sampling_steps)))
        timesteps = timesteps + 1
        timesteps_prev = torch.cat([torch.tensor([0]), timesteps[:-1]])

        x_i = torch.randn(
            (batch_size, self.channels, self.image_size, self.image_size),
            device=device,
        )
        imgs = [x_i]

        for i in trange(
            sampling_steps - 1,
            0,
            -1,
            desc="DDIM Sampling...",
            leave=False,
            dynamic_ncols=True,
        ):
            t = torch.full((len(x_i),), timesteps[i], dtype=torch.long, device=device)

            alpha_t = self.alphabar_t[timesteps[i]]
            alpha_t_prev = self.alphabar_t[timesteps_prev[i]]

            eps = self.net(x_i, t)

            sigma_t = self.ddim_eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            )
            epsilon_t = torch.randn_like(x_i)

            x_i = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_i
                + (
                    torch.sqrt(1 - alpha_t_prev - sigma_t**2)
                    - torch.sqrt((alpha_t_prev * (1 - alpha_t)) / alpha_t)
                )
                * eps
                + sigma_t * epsilon_t
            )
            imgs.append(x_i)
        
        x_i = unnormalize_to_0_1(x_i.clip(-1, 1))

        if return_all_steps:
            return x_i, unnormalize_to_0_1(torch.stack(imgs).clip(-1, 1))

        return x_i

    @torch.inference_mode()
    def cond_sample(
        self, classes, device, sampling_steps=50, cfg_scale=5.0, return_all_steps=False
    ):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        assert self.use_cond
        
        if sampling_steps < self.n_T:
            return self.ddim_cond_sample(
                classes, device, sampling_steps, cfg_scale, return_all_steps
            )
        
        y = torch.tensor(classes, device=device)
        x_i = torch.randn(
            (len(classes), self.channels, self.image_size, self.image_size),
            device=device,
        )

        imgs = [x_i]

        for i in trange(
            self.n_T,
            0,
            -1,
            desc=f"Sampling with cfg={cfg_scale}...",
            leave=False,
            dynamic_ncols=True,
        ):
            z = torch.randn_like(x_i) if i > 1 else 0

            t = torch.tensor([i] * len(x_i), device=device)
            eps = self.net.forward_with_cfg(x_i, t, y, cfg_scale)

            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

            imgs.append(x_i)

        x_i = unnormalize_to_0_1(x_i.clip(-1, 1))

        if return_all_steps:
            imgs = imgs[:: len(imgs) // 50]
            return x_i, unnormalize_to_0_1(torch.stack(imgs).clip(-1, 1))

        return x_i
    
    @torch.inference_mode()
    def ddim_cond_sample(
        self, classes, device, sampling_steps=50, cfg_scale=5.0, return_all_steps=False
    ):
        y = torch.tensor(classes, device=device)
        
        timesteps = torch.tensor(list(range(0, self.n_T, self.n_T // sampling_steps)))
        timesteps = timesteps + 1
        timesteps_prev = torch.cat([torch.tensor([0]), timesteps[:-1]])

        x_i = torch.randn(
            (len(classes), self.channels, self.image_size, self.image_size),
            device=device,
        )
        imgs = [x_i]

        for i in trange(
            sampling_steps - 1,
            0,
            -1,
            desc="DDIM Sampling...",
            leave=False,
            dynamic_ncols=True,
        ):
            t = torch.full((len(x_i),), timesteps[i], dtype=torch.long, device=device)

            alpha_t = self.alphabar_t[timesteps[i]]
            alpha_t_prev = self.alphabar_t[timesteps_prev[i]]

            eps = self.net.forward_with_cfg(x_i, t, y, cfg_scale)

            sigma_t = self.ddim_eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            )
            epsilon_t = torch.randn_like(x_i)

            x_i = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_i
                + (
                    torch.sqrt(1 - alpha_t_prev - sigma_t**2)
                    - torch.sqrt((alpha_t_prev * (1 - alpha_t)) / alpha_t)
                )
                * eps
                + sigma_t * epsilon_t
            )
            imgs.append(x_i)
        
        x_i = unnormalize_to_0_1(x_i.clip(-1, 1))

        if return_all_steps:
            return x_i, unnormalize_to_0_1(torch.stack(imgs).clip(-1, 1))

        return x_i

    def fid_sample(self, batch_size, device, cfg_scale=5.0):
        if self.use_cond:
            y = torch.randint(0, self.net.num_classes, (batch_size,), device=device)

            return self.cond_sample(y, device, cfg_scale=cfg_scale)
        else:
            return self.sample(batch_size, device)


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }
