from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


from vae_sam.runners.runner import SAMModule
from vae_sam.utils import add_tags
from pl_bolts.datamodules import CIFAR10DataModule


class MyLightningCLI(LightningCLI):
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
                "early_stopping.monitor": "Metrics/val/neg_elbo",
                "early_stopping.mode": "min",
                "early_stopping.patience": 5,
            }
        )

    def before_instantiate_classes(self) -> None:
        pass

        # self.config[self.subcommand].trainer.logger.init_args.tags = add_tags(
        #     self.config[self.subcommand]
        # )

    def before_fit(self):
        if isinstance(self.trainer.logger, WandbLogger) is True:
            # required as the parser cannot parse the "-" symbol
            self.trainer.logger.__dict__["_wandb_init"]["entity"] = "ima-vae"

            if self.config[self.subcommand].model.offline is True:
                self.trainer.logger.__dict__["_wandb_init"]["mode"] = "offline"
            else:
                self.trainer.logger.__dict__["_wandb_init"]["mode"] = "online"

            # todo: maybe set run in the CLI to false and call watch before?
            self.trainer.logger.watch(self.model, log="all", log_freq=250)


cli = MyLightningCLI(
    SAMModule,
    CIFAR10DataModule,
    save_config_callback=None,
    run=True,
    parser_kwargs={"parse_as_dict": False},
    trainer_defaults={
        "callbacks": [
            ModelCheckpoint(
                save_top_k=1,
                monitor="Metrics/val/neg_elbo",
                mode="min",
            ),
        ]
    },
)
