import subprocess
from os.path import dirname

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb

from vae_sam.models.ivae import iVAE
from vae_sam.models.utils import ActivationType
from vae_sam.models.utils import PriorType


class SAMModule(pl.LightningModule):
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        activation: ActivationType = "lrelu",
        latent_dim: int = 2,
        n_layers: int = 2,
        lr: float = 1e-3,
        n_classes: int = 10,
        dataset="synth",
        log_latents: bool = False,
        log_reconstruction: bool = False,
        prior: PriorType = "uniform",
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        decoder_var=0.000001,
        fix_prior: bool = True,
        beta=1.0,
        diag_posterior: bool = True,
        offline: bool = False,
        **kwargs,
    ):
        """

        :param offline: offline W&B run (sync at the end)
        :param diag_posterior: choose a diagonal posterior
        :param beta: beta of the beta-VAE
        :param fix_prior: fix (and not learn) prior distribution
        :param decoder_var: decoder variance
        :param prior_mean: prior mean
        :param prior_var: prior variance
        :param prior_alpha: beta prior alpha shape > 0
        :param prior_beta: beta prior beta shape > 0
        :param device: device to run on
        :param activation: activation function, any on 'lrelu', 'sigmoid', 'none'
        :param latent_dim: dimension of the latent space
        :param n_layers: number of layers
        :param lr: learning rate
        :param n_classes: number of classes
        :param log_latents: flag to log latents
        :param log_reconstruction: flag to log reconstructions
        :param prior: prior distribution name as string
        :param kwargs:
        """
        super().__init__()

        self.save_hyperparameters()

        self.model: iVAE = iVAE(
            latent_dim=latent_dim,
            data_dim=latent_dim,
            n_classes=n_classes,
            n_layers=n_layers,
            activation=activation,
            device=device,
            prior=prior,
            diag_posterior=diag_posterior,
            dataset=self.hparams.dataset,
            fix_prior=fix_prior,
            beta=beta,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            prior_mean=prior_mean,
            prior_var=prior_var,
            decoder_var=decoder_var,
        )

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.watch(self.model, log="all", log_freq=250)

    def forward(self, obs, labels):
        # in lightning, forward defines the prediction/inference actions
        return self.model(obs, labels)

    def training_step(self, batch, batch_idx):
        obs, labels, sources = batch
        neg_elbo, z_est, rec_loss, kl_loss, _, _, _ = self.model.neg_elbo(obs, labels)

        panel_name = "Metrics/train"
        self._log_metrics(kl_loss, neg_elbo, rec_loss, None, panel_name)

        return neg_elbo

    def _log_metrics(
        self, kl_loss, neg_elbo, rec_loss, latent_stat=None, panel_name: str = "Metrics"
    ):
        self.log(f"{panel_name}/neg_elbo", neg_elbo, on_epoch=True, on_step=False)
        self.log(f"{panel_name}/rec_loss", rec_loss, on_epoch=True, on_step=False)
        self.log(f"{panel_name}/kl_loss", kl_loss, on_epoch=True, on_step=False)
        if latent_stat is not None:
            self.log(
                f"{panel_name}/latent_statistics",
                latent_stat,
                on_epoch=True,
                on_step=False,
            )

    def validation_step(self, batch, batch_idx):
        obs, labels, sources = batch
        (
            neg_elbo,
            latent,
            rec_loss,
            kl_loss,
            latent_stat,
            reconstruction,
            encoding_mean,
        ) = self.model.neg_elbo(obs, labels, reconstruction=True, mean_latents=True)

        panel_name = "Metrics/val"
        self._log_metrics(kl_loss, neg_elbo, rec_loss, latent_stat, panel_name)

        if (
            self.current_epoch % 20 == 0
            or self.current_epoch == (self.trainer.max_epochs - 1)
        ) is True:
            self._log_latents(latent, panel_name)
            self._log_reconstruction(obs, reconstruction, panel_name)

        return neg_elbo

    def test_step(self, batch, batch_idx):
        obs, labels, sources = batch
        (
            neg_elbo,
            latent,
            rec_loss,
            kl_loss,
            latent_stat,
            reconstruction,
            _,
        ) = self.model.neg_elbo(obs, labels, reconstruction=True)

        panel_name = "Metrics/test"
        self._log_metrics(kl_loss, neg_elbo, rec_loss, latent_stat, panel_name)

        self._log_latents(latent, panel_name)
        self._log_reconstruction(obs, reconstruction, panel_name)

    def _log_reconstruction(self, obs, rec, panel_name, max_img_num: int = 5):
        if (
            rec is not None
            and self.hparams.log_reconstruction is True
            and isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True
        ):
            wandb_logger = self.logger.experiment
            # not images
            if len(obs.shape) == 2:
                table = wandb.Table(
                    columns=[f"dim={i}" for i in range(self.hparams.latent_dim)]
                )
                imgs = []
                for i in range(self.hparams.latent_dim):
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    imgs.append(
                        wandb.Image(
                            ax.scatter(
                                obs[:, i], rec[:, i], label=[f"obs_{i}", f"rec_{i}"]
                            )
                        )
                    )

                table.add_data(*imgs)

            # images
            else:
                table = wandb.Table(columns=["Observations", "Reconstruction"])
                for i in range(max_img_num):
                    table.add_data(wandb.Image(obs[i, :]), wandb.Image(rec[i, :]))

            wandb_logger.log({f"{panel_name}/reconstructions": table})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _log_latents(self, latent, panel_name):

        if (
            self.logger is not None
            and self.hparams.log_latents is True
            and isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True
        ):

            wandb_logger = self.logger.experiment
            table = wandb.Table(
                columns=["Idx"]
                + [f"latent_{i}" for i in range(self.hparams.latent_dim)]
            )
            for row in range(self.hparams.latent_dim - 1):
                imgs = [row]
                imgs += [None] * (row + 1)
                for col in range(row + 1, self.hparams.latent_dim):
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    imgs.append(
                        wandb.Image(
                            ax.scatter(
                                latent[:, row],
                                latent[:, col],
                                label=[f"latent_{row}", f"latent_{col}"],
                            )
                        )
                    )

                table.add_data(*imgs)

            wandb_logger.log({f"{panel_name}/latents": table})

    def on_fit_end(self) -> None:
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            if self.hparams.offline is True:
                # Syncing W&B at the end
                # 1. save sync dir (after marking a run finished, the W&B object changes (is teared down?)
                sync_dir = dirname(self.logger.experiment.dir)
                # 2. mark run complete
                wandb.finish()
                # 3. call the sync command for the run directory
                subprocess.check_call(["wandb", "sync", sync_dir])
