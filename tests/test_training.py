import torch
from pytorch_lightning.trainer import Trainer

from vae_sam.models.vae import VAE
import pytest


@pytest.mark.parametrize("enc_var", [None, 1.0])
def test_fast_dev_run(cifar10dm, enc_var):
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    model = VAE(enc_var=enc_var)
    trainer = Trainer(
        fast_dev_run=True,
        gpus=AVAIL_GPUS,
    )
    trainer.fit(model=model, datamodule=cifar10dm)


@pytest.mark.parametrize("enc_var", [None, 1.0])
def test_fast_dev_run_sampled(cifar10dm, enc_var):
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    model = VAE(val_num_samples=4, enc_var=enc_var)
    trainer = Trainer(
        fast_dev_run=True,
        gpus=AVAIL_GPUS,
    )
    trainer.fit(model=model, datamodule=cifar10dm)


def test_fast_dev_run_rae(cifar10dm):
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    enc_var = 1.0
    model = VAE(enc_var=enc_var, rae_update=True)
    trainer = Trainer(
        fast_dev_run=True,
        gpus=AVAIL_GPUS,
    )
    trainer.fit(model=model, datamodule=cifar10dm)
