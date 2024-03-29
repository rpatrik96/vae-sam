from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger

from vae_sam.models.vae import VAE
from vae_sam.utils import add_tags


class SAMLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--notes",
            type=str,
            default=None,
            help="Notes for the run on Weights and Biases",
        )
        parser.add_argument(
            "--tags",
            type=str,
            nargs="*",  # 0 or more values expected => creates a list
            default=None,
            help="Tags for the run on Weights and Biases",
        )

    def before_instantiate_classes(self) -> None:
        if (
            self.config[self.subcommand].trainer.logger.class_path
            == "pytorch_lightning.loggers.WandbLogger"
        ):
            self.config[self.subcommand].trainer.logger.init_args.tags = add_tags(
                self.config[self.subcommand]
            )

            if self.config[self.subcommand].model.offline is True:
                self.config[self.subcommand].trainer.logger.init_args.mode = "offline"
            else:
                self.config[self.subcommand].trainer.logger.init_args.mode = "online"

    def before_fit(self):
        if isinstance(self.trainer.logger, WandbLogger) is True:
            if self.config[self.subcommand].model.offline is True:
                self.trainer.logger.__dict__["_wandb_init"]["mode"] = "offline"
            else:
                self.trainer.logger.__dict__["_wandb_init"]["mode"] = "online"


cli = SAMLightningCLI(
    VAE,
    CIFAR10DataModule,
    save_config_callback=None,
    run=True,
    trainer_defaults={
        "callbacks": [
            ModelCheckpoint(
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            ),
        ]
    },
)
