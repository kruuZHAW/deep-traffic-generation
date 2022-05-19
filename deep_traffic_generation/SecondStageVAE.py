from deep_traffic_generation.core import FCN
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import torch


class VAE(pl.LightningModule):
    def __init__(
        self,
        input_dim=128,
        latent_dim=128,
        h_dims=[
            128
        ],  # gives dimensions of hidden layers + dimensions of output of encoder
        h_activ=nn.ReLU(),
        batch_norm=True,
    ):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = FCN(
            input_dim=input_dim,
            out_dim=h_dims[-1],
            h_dims=h_dims[:-1],
            batch_norm=batch_norm,
            h_activ=h_activ,
            dropout=0,
        )

        self.decoder = FCN(
            input_dim=latent_dim,
            out_dim=input_dim,
            h_dims=h_dims[::-1],
            batch_norm=batch_norm,
            h_activ=h_activ,
            dropout=0,
        )

        # distribution parameters
        self.fc_mu = nn.Linear(h_dims[-1], latent_dim)
        self.fc_var = nn.Linear(h_dims[-1], latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.out_activ = nn.Tanh()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=200, gamma=0.5
        )
        return {"optimizer": optimizer, "scheduler": scheduler}

    def gen_loss(
        self, x: torch.Tensor, x_hat: torch.Tensor, gamma: torch.Tensor
    ):
        """Computes generation loss in TwoStages VAE Model
        Args :
            x : input data
            x_hat : reconstructed data
            gamma : decoder std (scalar as every distribution in the decoder has the same std)

        To use it within the learning : take the sum and divide by the batch size
        """
        HALF_LOG_TWO_PI = 0.91893

        loggamma = torch.log(gamma)
        return (
            torch.square((x - x_hat) / gamma) / 2.0 + loggamma + HALF_LOG_TWO_PI
        )

    def kl_loss(self, mu: torch.Tensor, std: torch.Tensor):
        """Computes close form of KL for gaussian distributions
        Args :
            mu : encoder means
            std : encoder stds

        To use it within the learning : take the sum and divide by the batch size
        """
        logstd = torch.log(std)
        return (torch.square(mu) + torch.square(std) - 2 * logstd - 1) / 2.0

    def training_step(self, batch, batch_idx):
        x = batch
        batch_size = x.shape[0]
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        self.loc = mu
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.out_activ(self.decoder(z))

        # Scale :
        self.calculated_scale = torch.Tensor([torch.sqrt(F.mse_loss(x, x_hat))])

        # reconstruction loss
        recon_loss = (
            torch.sum(self.gen_loss(x, x_hat, torch.exp(self.log_scale)))
            / batch_size
        )

        # kl
        kl = torch.sum(self.kl_loss(mu, std)) / batch_size

        elbo = kl + recon_loss
        elbo = elbo

        self.log_dict(
            {
                "elbo": elbo,
                "kl": kl,
                "recon_loss": recon_loss,
            }
        )

        return elbo
