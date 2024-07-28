import torch
from torchvision.utils import make_grid, save_image
from accelerate import Accelerator

from tqdm.auto import tqdm
from contextlib import contextmanager
from ema import LitEma
from loss_fn import LPIPSWithDiscriminator

import os
from loguru import logger
from pathlib import Path
import datetime
import wandb
from modules.utils import unnormalize_to_0_1
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from einops import rearrange, repeat
import numpy as np


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class GANTrainer:
    def __init__(
        self,
        config,
        accelerator: Accelerator,
        generator,
        discriminator,
        gen_optimizer,
        disc_optimizer,
        loss_fn: LPIPSWithDiscriminator,
        datamodule,
        gen_lr_scheduler=None,
        disc_lr_scheduler=None,
        # hyperparameters
        batch_size=1,
        grad_accumulate_every=1,
        max_grad_norm=torch.inf,
        n_steps=1,
        use_ema=True,
        ema_decay=0.9999,
        # misc
        log_dir=None,
        sample_interval=1000,
        validate_interval=1000,
        logger_kwargs={},
        **kwargs,
    ):
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        self.datamodule = datamodule
        self.generator, self.opt_g, self.discriminator, self.opt_d, self.dl = (
            accelerator.prepare(
                generator,
                gen_optimizer,
                discriminator,
                disc_optimizer,
                datamodule.get_train_dataloader(),
            )
        )
        self.loss_fn = loss_fn.to(self.device)
        self.gen_lr_scheduler = gen_lr_scheduler
        self.disc_lr_scheduler = disc_lr_scheduler

        self.batch_size = batch_size
        self.grad_accumulate_every = grad_accumulate_every
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps

        self.sample_interval = sample_interval
        self.validate_interval = validate_interval

        self.use_ema = use_ema
        if self.use_ema:
            self.generator_ema = LitEma(self.generator, decay=ema_decay)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_v3 = InceptionV3([block_idx]).to(self.device)

        self.logger_kwargs = logger_kwargs
        self.step = 0

        current_datetime = datetime.datetime.now()
        date_string = current_datetime.strftime("%Y-%m-%d")
        time_string = current_datetime.strftime("%H-%M-%S")
        self.log_dir = Path(f"{log_dir}/{date_string}/{time_string}")
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_paths = []

    @contextmanager
    def ema_scope(self):
        if self.use_ema:
            self.generator_ema.store(self.generator.parameters())
            self.generator_ema.copy_to(self.generator)
        try:
            yield None
        finally:
            if self.use_ema:
                self.generator_ema.restore(self.generator.parameters())

    def save_ckpt(self):
        state_dict = {
            "config": self.config,
            "generator": self.accelerator.get_state_dict(self.generator),
            "discriminator": self.accelerator.get_state_dict(self.discriminator),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "step": self.step,
        }
        if self.accelerator.scaler is not None:
            state_dict["scaler"] = self.accelerator.scaler.state_dict()
        if self.use_ema:
            state_dict["generator_ema"] = self.generator_ema.state_dict()

        ckpt_path = self.log_dir / f"model-step{self.step}.ckpt"
        torch.save(state_dict, ckpt_path)
        logger.info(f"Saved checkpoint at {ckpt_path}")
        self.ckpt_paths.append(ckpt_path)

    def load_ckpt(self, ckpt_path):
        accelerator = self.accelerator
        device = accelerator.device

        logger.info(f"Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

        generator = accelerator.unwrap_model(self.generator)
        generator.load_state_dict(state_dict["generator"])

        discriminator = accelerator.unwrap_model(self.discriminator)
        discriminator.load_state_dict(state_dict["discriminator"])

        self.opt_g.load_state_dict(state_dict["opt_g"])
        self.opt_d.load_state_dict(state_dict["opt_d"])
        self.step = state_dict["step"]
        if accelerator.scaler is not None:
            accelerator.scaler.load_state_dict(state_dict["scalar"])
        if self.use_ema:
            self.generator_ema.load_state_dict(state_dict["generator_ema"])
        
        self.step = state_dict["step"]
    
    def train(self):
        self.accelerator.init_trackers(**self.logger_kwargs, config=self.config)

        with tqdm(range(self.n_steps), desc="Training", dynamic_ncols=True) as pbar, self.accelerator.autocast():
            pbar.n = self.step
            pbar.last_print_n = self.step

            for _ in pbar:
                # Update Discriminator
                self.opt_d.zero_grad()

                total_d_loss = 0.0
                total_loss_dict_d = {}
                for _ in range(self.grad_accumulate_every):
                    data = next(self.dl)
                    x, _ = data
                    x = x.to(self.device)
                    
                    with self.accelerator.autocast():
                        x_recon, _ = self.generator(x)
                        loss_d, loss_dict_d = self.loss_fn.discriminator_step(
                            x, x_recon.clone().detach(), self.discriminator, self.step
                        )
                    loss_d = loss_d / self.grad_accumulate_every
                    total_d_loss += loss_d.item()

                    for k, v in loss_dict_d.items():
                        if "logits" in k:
                            if k not in total_loss_dict_d:
                                total_loss_dict_d[k] = []
                            total_loss_dict_d[k].extend(v)
                        else:
                            if k not in total_loss_dict_d:
                                total_loss_dict_d[k] = 0.0
                            total_loss_dict_d[k] += v / self.grad_accumulate_every

                    self.accelerator.backward(loss_d)

                self.accelerator.clip_grad_norm_(
                    self.discriminator.parameters(), self.max_grad_norm
                )
                self.opt_d.step()

                # Update Generator
                self.opt_g.zero_grad()

                total_g_loss = 0.0
                total_loss_dict_g = {}
                for _ in range(self.grad_accumulate_every):
                    data = next(self.dl)
                    x, _ = data
                    x = x.to(self.device)

                    with self.accelerator.autocast():
                        x_recon, z = self.generator(x)
                        loss_g, loss_dict_g = self.loss_fn.generator_step(
                            x,
                            x_recon,
                            z,
                            self.discriminator,
                            self.generator.get_last_layer(),
                            self.step,
                        )
                    loss_g = loss_g / self.grad_accumulate_every
                    total_g_loss += loss_g.item()

                    for k, v in loss_dict_g.items():
                        if k not in total_loss_dict_g:
                            total_loss_dict_g[k] = 0.0
                        total_loss_dict_g[k] += v / self.grad_accumulate_every

                    self.accelerator.backward(loss_g)

                self.accelerator.clip_grad_norm_(
                    self.generator.parameters(), self.max_grad_norm
                )
                self.opt_g.step()
                if self.use_ema:
                    self.generator_ema(self.generator)

                self.step += 1

                lr_g = self.opt_g.param_groups[0]["lr"]
                lr_d = self.opt_d.param_groups[0]["lr"]

                pbar.set_postfix(
                    {
                        "loss_g": total_g_loss,
                        "rec_loss": total_loss_dict_g["rec_loss"].item(),
                        "var_loss": total_loss_dict_g["var_loss"].item(),
                        "adv_loss": total_loss_dict_g["g_loss"].item(),
                        "loss_d": total_d_loss,
                        "lr_g": lr_g,
                        "lr_d": lr_d,
                    }
                )
                total_loss_dict_g = {
                    f"train/gen_{k}": v for k, v in total_loss_dict_g.items()
                }
                total_loss_dict_d = {
                    f"train/disc_{k}": v for k, v in total_loss_dict_d.items()
                }

                for k in total_loss_dict_d.keys():
                    if "logits" in k:
                        total_loss_dict_d[k] = wandb.Histogram(total_loss_dict_d[k])
                self.accelerator.log(
                    {
                        "loss_g": total_g_loss,
                        "loss_d": total_d_loss,
                        "lr_g": lr_g,
                        "lr_d": lr_d,
                        "train/z_std": z.detach().flatten().std(),
                        "train/z_scale": self.generator.scale.item(),
                        **total_loss_dict_g,
                        **total_loss_dict_d,
                    },
                    self.step,
                )

                if self.step % 25 == 0:
                    self.log_gradients()

                if self.gen_lr_scheduler is not None:
                    self.gen_lr_scheduler.step()
                if self.disc_lr_scheduler is not None:
                    self.disc_lr_scheduler.step()

                if self.step % self.sample_interval == 0:
                    self.log_samples(x, x_recon)

                if self.step % self.validate_interval == 0:
                    with self.ema_scope():
                        self.validate()
                    self.save_ckpt()

        self.accelerator.end_training()

    def log_gradients(self):
        generator_grad = {
            f"generator_grad/{n}": p.grad.norm().item()
            for n, p in self.generator.named_parameters()
            if p.grad is not None
        }

        discriminator_grad = {
            f"discriminator_grad/{n}": p.grad.norm().item()
            for n, p in self.discriminator.named_parameters()
            if p.grad is not None
        }

        self.accelerator.log({**generator_grad, **discriminator_grad}, self.step)

    def log_samples(self, x, x_recon, split="train"):
        x, x_recon = unnormalize_to_0_1(x), unnormalize_to_0_1(x_recon.clip(-1, 1))
        x, x_recon = make_grid(x), make_grid(x_recon)
        logging_x, logging_x_recon = wandb.Image(
            x, caption="Ground-Truth"
        ), wandb.Image(x_recon, caption="Reconstructions")
        self.accelerator.log(
            {
                f"{split}/samples": [logging_x, logging_x_recon],
            },
            self.step,
        )

    def calculate_inception_features(self, samples):
        if samples.shape[1] == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    @torch.no_grad()
    def validate(self):
        self.generator.val()
        self.discriminator.val()
        dl = self.datamodule.get_val_dataloader()

        total_loss_dict_g = {}
        total_loss_dict_d = {}

        stacked_real_features = []
        stacked_fake_features = []

        num_batch = 0
        
        with tqdm(dl, desc="Validating", dynamic_ncols=True) as pbar:
            for i, batch in enumerate(pbar):
                x, _ = batch
                x = x.to(self.device)
                
                with self.accelerator.autocast():
                    x_recon, z = self.generator(x)
                    _, loss_dict_g = self.loss_fn.generator_step(
                        x, x_recon, z, self.discriminator, None, self.step
                    )

                for k, v in loss_dict_g.items():
                    if k not in total_loss_dict_g:
                        total_loss_dict_g[k] = 0.0
                    total_loss_dict_g[k] += v

                with self.accelerator.autocast():
                    _, loss_dict_d = self.loss_fn.discriminator_step(
                        x, x_recon, self.discriminator, self.step
                    )
                
                for k, v in loss_dict_d.items():
                    if "logits" in k:
                        if k not in total_loss_dict_d:
                            total_loss_dict_d[k] = []
                        total_loss_dict_d[k].extend(v)
                    else:
                        if k not in total_loss_dict_d:
                            total_loss_dict_d[k] = 0.0
                        total_loss_dict_d[k] += v

                real_features = self.calculate_inception_features(unnormalize_to_0_1(x))
                fake_features = self.calculate_inception_features(
                    unnormalize_to_0_1(x_recon.clip(-1, 1))
                )
                stacked_real_features.append(real_features)
                stacked_fake_features.append(fake_features)

                num_batch += 1
                if i == 42:
                    self.log_samples(x, x_recon, split="val")

        total_loss_dict_g = {
            f"val/gen_{k}": v / num_batch for k, v in total_loss_dict_g.items()
        }
        total_loss_dict_d = {
            f"val/disc_{k}": v / num_batch
            for k, v in total_loss_dict_d.items()
            if "logits" not in k
        }

        stacked_real_features = torch.cat(stacked_real_features, dim=0).cpu().float().numpy()
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().float().numpy()
        m2, s2 = np.mean(stacked_real_features, axis=0), np.cov(
            stacked_real_features, rowvar=False
        )
        m1, s1 = np.mean(stacked_fake_features, axis=0), np.cov(
            stacked_fake_features, rowvar=False
        )

        fid = calculate_frechet_distance(m1, s1, m2, s2)
        logger.info(f"FID score at step {self.step}: {fid}")

        self.accelerator.log(
            {
                **total_loss_dict_g,
                **total_loss_dict_d,
                "val/z_std": z.detach().flatten().std(),
                "FID": fid,
            },
            self.step,
        )

        self.generator.train()
        self.discriminator.train()
