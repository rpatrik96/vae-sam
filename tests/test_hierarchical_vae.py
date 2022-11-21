import torch
from pl_bolts.datamodules import STL10DataModule

from vae_sam.models.hierarchical_vae import HierarchicalVAE

BATCH_SIZE = 8
NUM_CHANNEL_EXPANSION = 256
RGB_CHANNELS = 3


def test_hierarchical_vae_fwd():
    vae = HierarchicalVAE(
        RGB_CHANNELS, NUM_CHANNEL_EXPANSION, enc_out_dim=12, latent_dim=8
    )
    x = torch.randn((BATCH_SIZE, *STL10DataModule.dims))

    x_hat = vae(x)

    assert x.shape == x_hat.shape


def test_hierarchical_vae_run_step():
    vae = HierarchicalVAE(
        RGB_CHANNELS, NUM_CHANNEL_EXPANSION, enc_out_dim=12, latent_dim=8
    )
    x = torch.randn((BATCH_SIZE, *STL10DataModule.dims))

    z_hat, z_mu, y_hat, y_mu, x_hat, p_z, q_z, p_y, q_y = vae._run_step(x)


def test_hierarchical_vae_sam_update():
    vae = HierarchicalVAE(
        RGB_CHANNELS,
        NUM_CHANNEL_EXPANSION,
        enc_out_dim=12,
        latent_dim=8,
        sam_update=True,
    )
    x = torch.randn((BATCH_SIZE, *STL10DataModule.dims))

    z_hat, z_mu, y_hat, y_mu, x_hat, p_z, q_z, p_y, q_y = vae._run_step(x)
    recon_loss, recon_loss_sam = vae.rec_loss(y_mu, y_hat.shape, x, x_hat)

    assert recon_loss < recon_loss_sam
