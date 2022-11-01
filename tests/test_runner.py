from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from vae_sam.data import CIFAR10DataModule
from vae_sam.runners.runner import SAMModule


def test_runner_args(args):
    # add model specific args
    runner = SAMModule(**vars(args))

    # if the code reaches this point, then all required args are specified
    pass


def test_training_with_wandb_logging(args):
    args.fast_dev_run = True
    args.logger = WandbLogger(project="test", entity="vae_sam", offline=True)

    dict_args = vars(args)

    # init the trainer like this
    trainer = Trainer.from_argparse_args(args)

    # init the model with all the key-value pairs
    model = SAMModule(**dict_args)

    # datamodule
    dm = CIFAR10DataModule.from_argparse_args(args)

    # fit
    trainer.fit(model, datamodule=dm)
