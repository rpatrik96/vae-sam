from argparse import Namespace

import hydra
import torch.cuda
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything

from pl_bolts.datamodules import CIFAR10DataModule

from vae_sam.runners.runner import SAMModule


@hydra.main(config_path="../configs", config_name="trainer")
def main(cfg: DictConfig):
    seed_everything(cfg.seed_everything)

    if torch.cuda.is_available() is False:
        cfg.trainer.gpus = 0

    trainer = Trainer.from_argparse_args(Namespace(**cfg.trainer))
    model = SAMModule(**OmegaConf.to_container(cfg.model))
    dm = CIFAR10DataModule.from_argparse_args(Namespace(**cfg.data))

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
