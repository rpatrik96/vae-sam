import pytest
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.core.datamodule import LightningDataModule

from vae_sam.utils import CIFAR10_DIR


@pytest.fixture
def cifar10dm() -> LightningDataModule:
    return CIFAR10DataModule(data_dir=CIFAR10_DIR)
