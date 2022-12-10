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
