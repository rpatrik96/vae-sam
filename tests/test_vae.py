import torch
from pl_bolts.datamodules import CIFAR10DataModule

from vae_sam.models.vae import VAE

from math import sqrt
import pytest


def test_sam_run_step():
    batch_size = 128
    vae = VAE(sam_update=True, enc_var=0.001)
    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    z, z_mu, std, x_hat, _, _ = vae._run_step(x)
    rec_loss, rec_loss_sam, rec_loss_no_sam, _ = vae.rec_loss(z_mu, std, x, x_hat)

    assert rec_loss_no_sam < rec_loss_sam


def test_rae_run_step():
    batch_size = 128
    vae = VAE(enc_var=1.0, rae_update=True)
    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    z, z_mu, _, _, _, _ = vae._run_step(x)

    assert torch.allclose(z, z_mu)


def test_sam_linear_loss():
    batch_size = 128
    TOL = 5e-5
    loss = lambda n, m: (n - m).mean()
    vae = VAE(sam_update=True, rec_loss=loss)
    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    z, z_mu, log_var, x_hat, p, q = vae._run_step(x)

    dLdz, scale = vae.sam_step(x, z_mu, log_var)

    x_hat = vae.decoder(z_mu)
    x_hat_sam = vae.decoder(z_mu + scale * dLdz)

    dLdz = torch.autograd.grad(outputs=loss(x, x_hat), inputs=z_mu)[0].detach()

    dLdz_sam = torch.autograd.grad(outputs=loss(x, x_hat_sam), inputs=z_mu)[0].detach()

    assert (dLdz.abs() - dLdz_sam.abs()).mean().abs() < TOL


def test_sampling():
    batch_size = 8
    sample_shape = torch.Size([4])
    vae = VAE(sam_update=True)
    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    xx = vae.encoder(x)
    mu = vae.fc_mu(xx)
    std = vae.calc_enc_std(xx)

    _, _, z = vae.sample(mu, std, sample_shape)

    assert z.shape[:2] == torch.Size([sample_shape[0], batch_size])


def test_sampled_rec_loss():
    batch_size = 8
    sample_shape = torch.Size([4])
    vae = VAE(sam_update=True)
    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    xx = vae.encoder(x)
    mu = vae.fc_mu(xx)
    std = vae.calc_enc_std(xx)

    _, _, z = vae.sample(mu, std, sample_shape)

    torch.tensor([vae.rec_loss(mu, std, x, vae.decoder(zz)) for zz in z])


def test_sampled_rec_loss_step():
    batch_size = 8
    vae = VAE(sam_update=True, val_num_samples=4)
    x = torch.randn((batch_size, *CIFAR10DataModule.dims))
    y = None

    vae.step((x, y), batch_idx=0, sample_shape=vae.hparams.val_num_samples)


def test_fix_enc_var():
    batch_size = 8
    enc_var = 1.0
    vae = VAE(sam_update=True, enc_var=enc_var)

    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    x = vae.encoder(x)
    std = vae.calc_enc_std(x)

    assert std.requires_grad == False


def test_rae_kl():
    batch_size = 8
    enc_var = 1.0
    vae = VAE(sam_update=True, enc_var=enc_var, rae_update=True)

    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    xx = vae.encoder(x)
    z_mu = vae.fc_mu(xx)

    kl = vae.kl_loss(None, None, z_mu)

    assert kl == vae.hparams.kl_coeff * z_mu.norm(p=2.0) / 2.0


@pytest.mark.parametrize(
    "sam_update, rae_update", [(True, False), (False, True), (False, False)]
)
def test_loss_stats(sam_update, rae_update):
    batch_size = 8
    enc_var = 1.0
    sample_shape = torch.Size()
    vae = VAE(sam_update=sam_update, enc_var=enc_var, rae_update=rae_update)

    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    z, z_mu, std, x_hat, p, q = vae._run_step(x, sample_shape=sample_shape)

    logs = vae.rec_loss_stats(
        std=std, x=x, x_hat=x_hat, z_mu=z_mu, sample_shape=sample_shape
    )
    kl = vae.kl_loss(p, q, z_mu)
    logs = vae.loss_stats(kl, logs, x, z_mu)

    loss = logs["loss"]

    if sam_update is True and rae_update is False:
        assert loss == (kl + logs["recon_loss_sam"])
    elif sam_update is False and rae_update is True:
        assert loss == (kl + logs["grad_loss"] + logs["recon_loss_no_sam"])
    if sam_update is False and rae_update is False:
        assert loss == (kl + logs["recon_loss"])


def test_decoder_jacobian_grads():
    batch_size = 8
    vae = VAE(sam_update=False, rae_update=False)

    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    xx = vae.encoder(x)
    z_mu = vae.fc_mu(xx)

    assert vae._decoder_jacobian(x, z_mu).requires_grad == True


def test_decoder_jacobian_shape():
    batch_size = 8
    vae = VAE(sam_update=False)

    x = torch.randn((batch_size, *CIFAR10DataModule.dims))

    xx = vae.encoder(x)
    z_mu = vae.fc_mu(xx)

    assert vae._decoder_jacobian(x, z_mu).shape == torch.Size(
        [batch_size, vae.hparams.latent_dim]
    )


def test_grad_coeff_tying():
    latent_dim = 16
    vae = VAE(
        rae_update=True, tie_grad_coeff_sam=True, enc_var=1.0, latent_dim=latent_dim
    )

    assert vae.hparams.grad_coeff == sqrt(vae.hparams.enc_var * vae.hparams.latent_dim)
