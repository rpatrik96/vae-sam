import math
import subprocess
import urllib.parse
from argparse import ArgumentParser
from os.path import dirname
from typing import Optional, Union

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
        rho: float = 0.05,
        sam_update: bool = False,
        norm_p: float = 2.0,
        offline: bool = True,
        val_num_samples: Union[torch.Size, int] = torch.Size(),
        enc_var: Optional[float] = None,
        rae_update: bool = False,
        rec_loss=F.mse_loss,
        grad_coeff: float = 1.0,
        tie_grad_coeff_sam: bool = False,
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

        self._param_sanity_checks(enc_var, rae_update)

        self._setup_hparams()

        self._setup_networks(enc_type, first_conv, maxpool1)

    def _setup_hparams(self):
        self.save_hyperparameters()
        if not isinstance(self.hparams.val_num_samples, torch.Size):
            self.hparams.val_num_samples = torch.Size([self.hparams.val_num_samples])
        if self.hparams.rae_update is True and self.hparams.tie_grad_coeff_sam is True:
            if self.hparams.enc_var is None:
                raise ValueError(
                    f"For {self.hparams.tie_grad_coeff_sam=} enc_var must be set!"
                )
            self.hparams.grad_coeff = math.sqrt(
                self.hparams.enc_var * self.hparams.latent_dim
            )

    def _param_sanity_checks(self, enc_var, rae_update):
        if enc_var is not None:
            if isinstance(enc_var, float):
                if enc_var <= 0:
                    raise ValueError(f"{enc_var=}should be positive!")
        if enc_var is None and rae_update is True:
            raise ValueError(f"enc_var should be fixed when {rae_update=}!")

    def _setup_networks(self, enc_type, first_conv, maxpool1):
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
                self.hparams.latent_dim, self.hparams.input_height, first_conv, maxpool1
            )
        else:
            self.encoder = valid_encoders[enc_type]["enc"](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]["dec"](
                self.hparams.latent_dim, self.hparams.input_height, first_conv, maxpool1
            )
        self.fc_mu = nn.Linear(self.hparams.enc_out_dim, self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.enc_out_dim, self.hparams.latent_dim)

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
        std = self.calc_enc_std(x)

        p, q, z = self.sample(mu, std)
        return self.decoder(z)

    def calc_enc_std(self, x: torch.Tensor) -> torch.Tensor:
        if self.hparams.enc_var is None:
            std = self.fc_var(x).exp().sqrt()
        else:
            std = math.sqrt(self.hparams.enc_var) * torch.ones(
                (self.hparams.latent_dim,), device=x.device
            )

        return std

    def _run_step(self, x, sample_shape: torch.Size = torch.Size()):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        std = self.calc_enc_std(x)
        p, q, z = self.sample(mu, std, sample_shape=sample_shape)

        if self.hparams.rae_update is True:
            z = mu

        if sample_shape == torch.Size():
            x_hat = self.decoder(z)
        else:
            x_hat = [self.decoder(zz) for zz in z]
        return z, mu, std, x_hat, p, q

    def sample(
        self,
        mu: torch.Tensor,
        std: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> tuple[
        Optional[torch.distributions.Normal],
        Optional[torch.distributions.Normal],
        torch.tensor,
    ]:
        if self.hparams.rae_update is False:
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            q = torch.distributions.Normal(mu, std)
            z = q.rsample(sample_shape=sample_shape)
        else:
            p = q = None
            z = mu
        return p, q, z

    def step(self, batch, batch_idx, sample_shape: torch.Size = torch.Size()):
        x, y = batch
        z, z_mu, std, x_hat, p, q = self._run_step(x, sample_shape=sample_shape)

        logs = self.rec_loss_stats(sample_shape, std, x, x_hat, z_mu)
        kl = self.kl_loss(p, q, z_mu)

        logs = self.loss_stats(kl, logs, x, z_mu)

        return logs["loss"], logs

    def loss_stats(self, kl, logs, x, z_mu):
        if self.hparams.sam_update is False:
            loss = kl + logs["recon_loss"]
        elif self.hparams.rae_update is True:
            grad_loss = self.hparams.grad_coeff * self._decoder_jacobian(x, z_mu).norm(
                p=2.0
            )
            logs = {**logs, "grad_loss": grad_loss}

            loss = kl + grad_loss + logs["recon_loss_no_sam"]
        else:
            loss = kl + logs["recon_loss_sam"]
        logs = {
            **logs,
            "kl": kl,
            "loss": loss,
        }
        return logs

    def rec_loss_stats(
        self, sample_shape, std, x, x_hat, z_mu
    ) -> dict[str : torch.Tensor]:
        if sample_shape == torch.Size():
            rec_loss_vi, rec_loss_sam, rec_loss_no_sam, scale = self.rec_loss(
                z_mu, std, x, x_hat
            )
        else:
            rec_losses_tuple = [
                self.rec_loss(z_mu, std, x, x_hat_i) for x_hat_i in x_hat
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

        logs = {
            "recon_loss": rec_loss_vi,
            "recon_loss_sam": rec_loss_sam,
            "recon_loss_no_sam": rec_loss_no_sam,
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

        return logs

    def kl_loss(
        self,
        p: Optional[torch.distributions.Normal],
        q: Optional[torch.distributions.Normal],
        z_mu: torch.Tensor,
    ) -> torch.Tensor:
        if self.hparams.rae_update is False:
            kl = torch.distributions.kl_divergence(q, p).mean()
        else:
            kl = z_mu.norm(p=2.0) / 2.0
        return self.hparams.kl_coeff * kl

    def rec_loss(
        self,
        z_mu: torch.Tensor,
        std: torch.Tensor,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.hparams.sam_update is False:
            rec_loss_vi = F.mse_loss(x_hat, x, reduction="mean")
        else:
            with torch.no_grad():
                rec_loss_vi = F.mse_loss(x_hat, x, reduction="mean")

        if self.hparams.sam_update is True or self.training is False:
            if self.training is False:
                torch.set_grad_enabled(True)
                z_mu.requires_grad = True

            dLdz, scale = self.sam_step(x, z_mu, std)

            rec_loss_no_sam = F.mse_loss(
                self.decoder(z_mu), x, reduction="mean"
            ).detach()

            rec_loss_sam = F.mse_loss(
                self.decoder(z_mu + scale * std * dLdz),
                x,
                reduction="mean",
            )

            if self.training is False:
                torch.set_grad_enabled(False)
                rec_loss_sam = rec_loss_sam.detach()

        else:
            rec_loss_sam = rec_loss_no_sam = scale = torch.FloatTensor([-1.0])

        return rec_loss_vi, rec_loss_sam, rec_loss_no_sam, scale.detach().mean()

    def sam_step(
        self, x: torch.Tensor, z_mu: torch.Tensor, std: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dLdz = self._decoder_jacobian(x, z_mu).detach()

        dLdz = std.mean(dim=0, keepdim=True).detach() * dLdz
        scale = math.sqrt(self.hparams.latent_dim) / dLdz.norm(
            p=self.hparams.norm_p, dim=1, keepdim=True
        )

        return dLdz, scale

    def _decoder_jacobian(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(True):
            grad = torch.autograd.grad(
                outputs=self.hparams.rec_loss(self.decoder(z), x), inputs=z
            )[0]

        return grad

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
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)

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
