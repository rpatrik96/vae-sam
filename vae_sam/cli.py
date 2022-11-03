from pl_bolts.datamodules import CIFAR10DataModule, TinyCIFAR10DataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.cli import LightningCLI

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

        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults(
            {
                "early_stopping.monitor": "val_loss",
                "early_stopping.mode": "min",
                "early_stopping.patience": 5,
            }
        )

    def before_instantiate_classes(self) -> None:
        self.config[self.subcommand].trainer.logger.init_args.tags = add_tags(
            self.config[self.subcommand]
        )


cli = SAMLightningCLI(
    VAE,
    TinyCIFAR10DataModule,
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
