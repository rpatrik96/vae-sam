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


def test_sam_linear_loss():
    batch_size = 128
    TOL = 1e-5
    vae = VAE(sam_update=True)
    loss = lambda n, m: (n - m).mean()
    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    z, z_mu, x_hat, p, q = vae._run_step(x)

    dLdz, scale = vae.sam_step(x, z_mu, loss=loss)

    x_hat = vae.decoder(z_mu)
    x_hat_sam = vae.decoder(z_mu + scale * dLdz)

    dLdz = torch.autograd.grad(outputs=loss(x, x_hat), inputs=z_mu)[0].detach()

    dLdz_sam = torch.autograd.grad(outputs=loss(x, x_hat_sam), inputs=z_mu)[0].detach()

    assert (dLdz.mean() - dLdz_sam.mean()).abs() < TOL
