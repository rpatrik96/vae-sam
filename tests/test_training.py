import torch
from pytorch_lightning.trainer import Trainer

from vae_sam.models.vae import VAE


def test_fast_dev_run(cifar10dm):
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    model = VAE()
    trainer = Trainer(
        fast_dev_run=True,
        gpus=AVAIL_GPUS,
    )
    trainer.fit(model=model, datamodule=cifar10dm)


def test_fast_dev_run_sampled(cifar10dm):
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    model = VAE(val_num_samples=4)
    trainer = Trainer(
        fast_dev_run=True,
        gpus=AVAIL_GPUS,
    )
    trainer.fit(model=model, datamodule=cifar10dm)
