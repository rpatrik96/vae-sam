import torch
from pl_bolts.datamodules import CIFAR10DataModule

from vae_sam.models.vae import VAE


def test_sam_update():
    batch_size = 8
    vae = VAE(sam_update=True)
    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    z, z_mu, x_hat, p, q = vae._run_step(x)
    rec_loss, rec_loss_sam, rec_loss_no_sam = vae.rec_loss(z_mu, x, x_hat)

    assert rec_loss_no_sam < rec_loss_sam


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


def test_alpha_sam():
    batch_size = 128
    TOL = 2e-7
    vae = VAE(sam_update=True)

    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    z, z_mu, x_hat, p, q = vae._run_step(x)

    dLdz, scale = vae.sam_step(x, z_mu)

    # SGD
    vae.hparams.alpha = 0.0
    grad = vae.assemble_alpha_sam_grad(dLdz, scale)
    assert (grad + dLdz).mean().abs() < TOL

    # SAM
    vae.hparams.alpha = 1.0
    grad = vae.assemble_alpha_sam_grad(dLdz, scale)
    assert (grad - scale * dLdz).mean().abs() < TOL


def test_sampling():
    batch_size = 8
    sample_shape = torch.Size([4])
    vae = VAE(sam_update=True)
    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    x = vae.encoder(x)
    mu = vae.fc_mu(x)
    log_var = vae.fc_var(x)

    _, _, z = vae.sample(mu, log_var, sample_shape)

    assert z.shape[:2] == torch.Size([sample_shape[0], batch_size])


def test_sampled_rec_loss():
    batch_size = 8
    sample_shape = torch.Size([4])
    vae = VAE(sam_update=True)
    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    xx = vae.encoder(x)
    mu = vae.fc_mu(xx)
    log_var = vae.fc_var(xx)

    _, _, z = vae.sample(mu, log_var, sample_shape)

    torch.tensor([vae.rec_loss(mu, x, vae.decoder(zz)) for zz in z])


def test_sampled_rec_loss_step():
    batch_size = 8
    vae = VAE(sam_update=True, val_num_samples=4)
    x = torch.randn((batch_size, *CIFAR10DataModule.dims))
    y = None

    vae.step((x, y), batch_idx=0, sample_shape=vae.hparams.val_num_samples)
