import subprocess
from argparse import ArgumentParser
from os.path import dirname

import pytorch_lightning as pl
import torch
import wandb
from compressai.layers import GDN
from compressai.models.utils import conv, deconv
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F


class HierarchicalVAE(LightningModule):
    """Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
    """

    def __init__(
        self,
        N,
        M,
        input_height: int = 32,
        enc_out_dim: int = 512,
        kl_coeff_y: float = 0.1,
        kl_coeff_z: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        top_enc_out_dim=512,
        top_latent_dim=256,
        rho=1.0,
        sam_update=False,
        norm_p=2.0,
        offline=True,
        sam_validation=True,
        **kwargs,
    ):
        """
        Args:
             N (int): Number of channels
            M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
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

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, 2 * N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, 2 * M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.N = int(N)
        self.M = int(M)

    def forward(self, x):
        # level one
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        # level two
        z_params = z.view(z.shape[0], -1)
        z_mu, z_log_var = (
            z_params[:, : z_params.shape[1] // 2],
            z_params[:, z_params.shape[1] // 2 :],
        )

        p_z, q_z, z_hat = self.sample(z_mu, z_log_var)

        # in the paper, the conditional part is determining the variance of a 0 mean Gaussian,
        # if we want to include the means as well, then we can use the conditioner from the IMA VAE repo

        z_hat_shape = list(z.shape)
        z_hat_shape[1] = z_hat_shape[1] // 2

        scales_hat = self.h_s(z_hat.view(*z_hat_shape))

        # level one
        y_params = scales_hat.view(scales_hat.shape[0], -1)
        y_params = F.adaptive_avg_pool1d(y_params, 2 * x.shape[-1] ** 2)
        y_mu, y_log_var = (
            y_params[:, : y_params.shape[1] // 2],
            y_params[:, y_params.shape[1] // 2 :],
        )

        p_y, q_y, y_hat = self.sample(y_mu, y_log_var)

        # reconstruction
        x_hat = self.g_s(y_hat.view(*y.shape))

        return x_hat

    def _run_step(self, x):
        # level one
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        # level two
        z_params = z.view(z.shape[0], -1)
        z_mu, z_log_var = (
            z_params[:, : z_params.shape[1] // 2],
            z_params[:, z_params.shape[1] // 2 :],
        )

        p_z, q_z, z_hat = self.sample(z_mu, z_log_var)

        # in the paper, the conditional part is determining the variance of a 0 mean Gaussian,
        # if we want to include the means as well, then we can use the conditioner from the IMA VAE repo

        z_hat_shape = list(z.shape)
        z_hat_shape[1] = z_hat_shape[1] // 2

        scales_hat = self.h_s(z_hat.view(*z_hat_shape))

        # level one
        y_params = scales_hat.view(scales_hat.shape[0], -1)
        y_params = F.adaptive_avg_pool1d(y_params, 2 * x.shape[-1] ** 2)
        y_mu, y_log_var = (
            y_params[:, : y_params.shape[1] // 2],
            y_params[:, y_params.shape[1] // 2 :],
        )

        p_y, q_y, y_hat = self.sample(y_mu, y_log_var)

        y_hat = y_hat.view(*y.shape)
        # reconstruction
        x_hat = self.g_s(y_hat)
        return z_hat, z_mu, y_hat, y_mu, x_hat, p_z, q_z, p_y, q_y

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z_hat, z_mu, y_hat, y_mu, x_hat, p_z, q_z, p_y, q_y = self._run_step(x)

        recon_loss, recon_loss_sam = self.rec_loss(y_mu, y_shape, x, x_hat)

        kl_z = (
            self.hparams.kl_coeff_z * torch.distributions.kl_divergence(q_z, p_z).mean()
        )
        kl_y = (
            self.hparams.kl_coeff_y * torch.distributions.kl_divergence(q_y, p_y).mean()
        )
        kl = kl_y + kl_z

        if self.hparams.sam_update is False:
            loss = kl + recon_loss
        else:
            loss = kl + recon_loss_sam

        logs = {
            "recon_loss": recon_loss,
            "recon_loss_sam": recon_loss_sam,
            "kl_z": kl_z,
            "kl_y": kl_y,
            "loss": loss,
        }
        return loss, logs

    def rec_loss(
        self, y_mu: torch.Tensor, y_shape, x: torch.Tensor, x_hat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hparams.sam_update is False:
            recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        else:
            with torch.no_grad():
                recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        if self.hparams.sam_update is True and self.hparams.sam_validation is True:

            if self.training is False:
                torch.set_grad_enabled(True)
                y_mu.requires_grad = True

            y_mu_reshaped = y_mu.reshape(y_shape)
            dLdy = torch.autograd.grad(
                outputs=F.mse_loss(self.g_s(y_mu_reshaped), x), inputs=y_mu_reshaped
            )[0].detach()
            scale = self.hparams.rho / dLdy.view(y_mu.shape).norm(
                p=self.hparams.norm_p, dim=1, keepdim=True
            )
            recon_loss_sam = F.mse_loss(
                self.g_s((y_mu + scale * dLdy.view(y_mu.shape)).view(y_shape)),
                x,
                reduction="mean",
            )
            if self.training is False:
                torch.set_grad_enabled(False)
                recon_loss_sam = recon_loss_sam.detach()

        else:
            recon_loss_sam = -1.0

        return recon_loss, recon_loss_sam

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
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
