import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.lpips import LPIPS
import wandb


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        disc_factor=1.0,
        disc_weight=1.0,
        pixel_weight=1.0,
        perceptual_weight=1.0,
        var_weight=1.0,
    ):
        super().__init__()
        self.disc_start = disc_start
        self.disc_factor = disc_factor
        self.disc_weight = disc_weight
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.var_weight = var_weight
        self.lpips = LPIPS().eval()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight

    def generator_step(self, x, x_recon, z, discriminator, last_layer, step):
        rec_loss = torch.abs(x_recon - x).mean([1, 2, 3])
        if self.perceptual_weight > 0:
            perceptual_loss = self.lpips(x_recon, x).mean([1, 2, 3])
            rec_loss = rec_loss + self.perceptual_weight * perceptual_loss
        
        rec_loss = rec_loss.mean()

        var_loss = torch.abs(z.flatten().std() - 1.0)
        var_loss = var_loss * self.var_weight  
        
        logits_fake = discriminator(x_recon)
        g_loss = -torch.mean(logits_fake)
        if last_layer is not None:
            d_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer)
        else:
            d_weight = 0.0
        disc_factor = adopt_weight(self.disc_factor, step, self.disc_start)
        g_loss = g_loss * d_weight * disc_factor

        loss = rec_loss + var_loss + g_loss
        return loss, {
            "rec_loss": rec_loss,
            "var_loss": var_loss,
            "disc_factor": disc_factor,
            "g_loss": g_loss,
            "loss": loss,
        }

    def discriminator_step(self, x, x_recon, discriminator, step):
        gen_logits = discriminator(x_recon.detach())
        loss_gen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

        real_logits = discriminator(x)
        loss_real = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()
        loss = (loss_gen + loss_real) / 2

        disc_factor = adopt_weight(self.disc_factor, step, self.disc_start)
        loss = loss * disc_factor

        gen_logits = gen_logits.cpu().detach().tolist()
        real_logits = real_logits.cpu().detach().tolist()

        return loss, {
            "loss_gen": loss_gen,
            "loss_real": loss_real,
            "loss": loss,
            "gen_logits": gen_logits,
            "real_logits": real_logits,
        }
