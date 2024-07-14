import hydra
from hydra.utils import instantiate
from icecream import install
from omegaconf import OmegaConf
from rich import print as rprint
from torchinfo import summary
from accelerate.utils import set_seed

install()


@hydra.main(config_path="./config", version_base=None)
def main(cfg):
    set_seed(cfg.hparams.seed)

    dm = instantiate(cfg.datamodule)(**cfg.hparams)
    generator = instantiate(cfg.generator)
    discriminator = instantiate(cfg.discriminator)
    rprint(summary(generator, verbose=0))
    rprint(summary(discriminator, verbose=0))
    optimizer_g = instantiate(cfg.optimizer_g)(generator.parameters())
    optimizer_d = instantiate(cfg.optimizer_d)(discriminator.parameters())
    lr_scheduler_g = None
    lr_scheduler_d = None
    if "lr_scheduler_g" in cfg:
        lr_scheduler_g = instantiate(cfg.lr_scheduler_g)(optimizer_g)
    if "lr_scheduler_d" in cfg:
        lr_scheduler_d = instantiate(cfg.lr_scheduler_d)(optimizer_d)
    loss_fn = instantiate(cfg.loss_fn)

    trainer = instantiate(cfg.trainer)(
        config=OmegaConf.to_container(cfg, resolve=True),
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=optimizer_g,
        disc_optimizer=optimizer_d,
        datamodule=dm,
        gen_lr_scheduler=lr_scheduler_g,
        disc_lr_scheduler=lr_scheduler_d,
        loss_fn=loss_fn,
        **cfg.hparams,
    )
    with open(f"{trainer.log_dir}/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    
    if "ckpt_path" in cfg:
        trainer.load_ckpt(cfg.ckpt_path)

    trainer.train()


if __name__ == "__main__":
    main()
