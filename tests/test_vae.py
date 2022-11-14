import torch
from pl_bolts.datamodules import CIFAR10DataModule

from vae_sam.models.vae import VAE


def test_sam_update():
    batch_size = 8
    vae = VAE(sam_update=True)
    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    z, z_mu, x_hat, p, q = vae._run_step(x)
    recon_loss, recon_loss_sam = vae.rec_loss(z_mu, x, x_hat)

    assert recon_loss < recon_loss_sam
