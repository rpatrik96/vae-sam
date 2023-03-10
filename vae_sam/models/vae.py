import subprocess
import urllib.parse
from argparse import ArgumentParser
from os.path import dirname

import pytorch_lightning as pl
import torch
import wandb
from pl_bolts import _HTTPS_AWS_HUB
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
import math

from functorch import vmap


class VAE(LightningModule):
    """Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
        # pretrained on stl10
        vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        "cifar10-resnet18": urllib.parse.urljoin(
            _HTTPS_AWS_HUB, "vae/vae-cifar10/checkpoints/epoch%3D89.ckpt"
        ),
        "stl10-resnet18": urllib.parse.urljoin(
            _HTTPS_AWS_HUB, "vae/vae-stl10/checkpoints/epoch%3D89.ckpt"
        ),
    }

    def __init__(
        self,
        input_height: int = 32,
        enc_type: str = "resnet18",
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        rho=0.05,
        sam_update=False,
        norm_p=2.0,
        offline=True,
        sam_validation=True,
        val_num_samples=torch.Size(),
        alpha=1.0,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super().__init__()

        self.save_hyperparameters()

        if not isinstance(self.hparams.val_num_samples, torch.Size):
            self.hparams.val_num_samples = torch.Size([self.hparams.val_num_samples])

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            "resnet18": {
                "enc": resnet18_encoder,
                "dec": resnet18_decoder,
            },
            "resnet50": {
                "enc": resnet50_encoder,
                "dec": resnet50_decoder,
            },
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder(
                self.latent_dim, self.input_height, first_conv, maxpool1
            )
        else:
            self.encoder = valid_encoders[enc_type]["enc"](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]["dec"](
                self.latent_dim, self.input_height, first_conv, maxpool1
            )

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + " not present in pretrained weights.")

        return self.load_from_checkpoint(
            VAE.pretrained_urls[checkpoint_name], strict=False
        )

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x, sample_shape: torch.Size = torch.Size()):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var, sample_shape=sample_shape)
        if sample_shape == torch.Size():
            x_hat = self.decoder(z)
        else:
            x_hat = [self.decoder(zz) for zz in z]
        return z, mu, log_var, x_hat, p, q

    def sample(self, mu, log_var, sample_shape: torch.Size = torch.Size()):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample(sample_shape=sample_shape)
        return p, q, z

    def step(self, batch, batch_idx, sample_shape: torch.Size = torch.Size()):
        x, y = batch
        z, z_mu, log_var, x_hat, p, q = self._run_step(x, sample_shape=sample_shape)

        if sample_shape == torch.Size():
            rec_loss_vi, rec_loss_sam, rec_loss_no_sam, scale = self.rec_loss(
                z_mu, log_var, x, x_hat
            )
        else:
            rec_losses_tuple = [
                self.rec_loss(z_mu, log_var, x, x_hat_i) for x_hat_i in x_hat
            ]
            rec_losses_vi = torch.tensor([r[0] for r in rec_losses_tuple])
            rec_losses_sam = torch.tensor([r[1] for r in rec_losses_tuple])
            rec_losses_no_sam = torch.tensor([r[2] for r in rec_losses_tuple])
            scales = torch.tensor([r[3].mean() for r in rec_losses_tuple])

            rec_loss_vi = rec_losses_vi.mean()
            rec_loss_std = rec_losses_vi.std(dim=0)

            rec_loss_sam = rec_losses_sam.mean()
            rec_loss_sam_std = rec_losses_sam.std(dim=0)

            rec_loss_no_sam = rec_losses_no_sam.mean()
            rec_loss_no_sam_std = rec_losses_no_sam.std(dim=0)

            scale = scales.mean()
            scale_std = scales.std(dim=0)

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        if self.hparams.sam_update is False:
            loss = kl + rec_loss_vi
        else:
            loss = kl + rec_loss_sam

        logs = {
            "recon_loss": rec_loss_vi,
            "recon_loss_sam": rec_loss_sam,
            "recon_loss_no_sam": rec_loss_no_sam,
            "kl": kl,
            "loss": loss,
            "scale": scale,
        }

        if sample_shape != torch.Size():
            logs = {
                **logs,
                "recon_loss_std": rec_loss_std,
                "recon_loss_sam_std": rec_loss_sam_std,
                "recon_loss_no_sam_std": rec_loss_no_sam_std,
                "scale_std": scale_std,
            }

        return loss, logs

    def rec_loss(
        self,
        z_mu: torch.Tensor,
        log_var: torch.Tensor,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.hparams.sam_update is False:
            rec_loss_vi = F.mse_loss(x_hat, x, reduction="mean")

            _, scale = self.sam_step(x, z_mu, log_var)
        else:
            with torch.no_grad():
                rec_loss_vi = F.mse_loss(x_hat, x, reduction="mean")

        if (
            self.hparams.sam_update is True
            and self.hparams.sam_validation is True
            or self.training is False
        ):
            if self.training is False:
                torch.set_grad_enabled(True)
                z_mu.requires_grad = True

            dLdz, scale = self.sam_step(x, z_mu, log_var)

            rec_loss_no_sam = F.mse_loss(
                self.decoder(z_mu), x, reduction="mean"
            ).detach()

            self.assemble_alpha_sam_grad(dLdz, scale)

            rec_loss_sam = F.mse_loss(
                self.decoder(
                    z_mu
                    + scale
                    * math.sqrt(self.hparams.alpha)
                    * log_var.exp().sqrt()
                    * dLdz
                ),
                x,
                reduction="mean",
            )

            if self.training is False:
                torch.set_grad_enabled(False)
                rec_loss_sam = rec_loss_sam.detach()

        else:
            rec_loss_sam = rec_loss_no_sam = -1.0

        return rec_loss_vi, rec_loss_sam, rec_loss_no_sam, scale.detach().mean()

    def assemble_alpha_sam_grad(self, dLdz, scale) -> torch.Tensor:
        if self.hparams.alpha == 1.0:
            return scale * dLdz
        elif self.hparams.alpha == 0.0:
            return -dLdz
        else:
            return dLdz + self.hparams.alpha * (scale * dLdz - dLdz)

    def sam_step(self, x, z_mu, log_var, loss=F.mse_loss):
        dLdz = torch.autograd.grad(outputs=loss(self.decoder(z_mu), x), inputs=z_mu)[
            0
        ].detach()

        dLdz = log_var.exp().sqrt().mean(dim=0, keepdim=True).detach() * dLdz
        scale = math.sqrt(self.hparams.latent_dim) / dLdz.norm(
            p=self.hparams.norm_p, dim=1, keepdim=True
        )

        return dLdz, scale

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(
            batch, batch_idx, sample_shape=self.hparams.val_num_samples
        )
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--enc_type", type=str, default="resnet18", help="resnet18/resnet50"
        )
        parser.add_argument("--first_conv", action="store_true")
        parser.add_argument("--maxpool1", action="store_true")
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets",
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

    def on_fit_end(self) -> None:
        if self.hparams.offline is True and isinstance(
            self.logger, pl.loggers.WandbLogger
        ):
            # Syncing W&B at the end
            # 1. save sync dir (after marking a run finished, the W&B object changes (is teared down?)
            sync_dir = dirname(self.logger.experiment.dir)
            # 2. mark run complete
            wandb.finish()
            # 3. call the sync command for the run directory
            subprocess.check_call(["wandb", "sync", sync_dir])
