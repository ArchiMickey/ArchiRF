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
    model = instantiate(cfg.model)
    rprint(summary(model, verbose=0))
    optimizer = instantiate(cfg.optimizer)(model.net.parameters())
    lr_scheduler = None
    if "lr_scheduler" in cfg:
        lr_scheduler = instantiate(cfg.lr_scheduler)(optimizer)
    
    trainer = instantiate(cfg.trainer)(
        config = OmegaConf.to_container(cfg, resolve=True),
        model=model,
        optimizer=optimizer,
        datamodule=dm,
        lr_scheduler=lr_scheduler,
        **cfg.hparams,
    )
    with open(f"{trainer.log_dir}/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    trainer.train()


if __name__ == "__main__":
    main()
