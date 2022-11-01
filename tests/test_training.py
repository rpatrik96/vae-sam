import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.trainer import Trainer

from vae_sam.models.vae import VAE
from vae_sam.utils import CIFAR10_DIR


def test_training():
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    dm = CIFAR10DataModule(data_dir=CIFAR10_DIR)
    model = VAE()
    trainer = Trainer(
        fast_dev_run=True,
        gpus=AVAIL_GPUS,
    )
    trainer.fit(model, dm)
