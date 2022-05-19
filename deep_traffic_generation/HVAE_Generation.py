from deep_traffic_generation.tchvae import TCHVAE
from deep_traffic_generation.core.datasets import TrafficDataset
from torch.distributions.distribution import Distribution
from torch.distributions import Independent, Normal

import torch
import numpy as np

from os import walk
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, TypedDict


class TwoStageVAE:
    def __init__(
        self,
        X: TrafficDataset,  # Traffic dataset used to train the first VAE stage
        sim_type: str = "generation",
    ):
        super().__init__()

        self.X = X
        self.sim_type = sim_type

        if sim_type not in ["generation", "reconstruction"]:
            raise ValueError(
                "Invalid sim type. Expected one of: %s"
                % ["generation", "reconstruction"]
            )

    def load(
        self,
        path: str,
        dataset_params: TypedDict,
    ):
        filenames = next(walk(path + "checkpoints/"), (None, None, []))[2]

        self.VAE = TCHVAE.load_from_checkpoint(
            path + "checkpoints/" + filenames[0],
            hparams_file=path + "/hparams.yaml",
            dataset_params=dataset_params,
        )
        self.VAE.eval()

    def latent_space(
        self, n_samples: int
    ):  # Gives the latent spaces of the VAE, and n_sample generated points with the prior

        # Display of latent variable z2 and the generation
        h2 = self.VAE.encoder_z2(self.X.data)
        q2 = self.VAE.lsr(h2)
        z2 = q2.rsample()

        p_z2 = self.VAE.lsr.get_prior()
        z2_gen = p_z2.sample(torch.Size([n_samples])).squeeze(1)

        z2_embeddings = np.concatenate(
            (z2.detach().numpy(), z2_gen.detach().numpy()), axis=0
        )

        # Display of latent variable z1 and the generation conditioned to the one of z2
        h1_x = self.VAE.encoder_z1_x(self.X.data)
        h1_z2 = self.VAE.encoder_z1_z2(z2)
        h1 = torch.cat((h1_x, h1_z2), 1)
        h1 = self.VAE.encoder_z1_joint(h1)
        q1 = self.VAE.q_z1(h1)
        z1 = q1.rsample()

        # z1_gen ~ p(z1|z2_gen)
        mean_pz1, scales_pz1 = self.VAE.p_z1(z2)
        p_z1 = Independent(Normal(mean_pz1, scales_pz1), 1)
        z1_gen = p_z1.rsample()

        z1_embeddings = np.concatenate(
            (z1.detach().numpy(), z1_gen.detach().numpy()), axis=0
        )

        return z2_embeddings, z1_embeddings

    def decode(self, latent):  # decode some given latent variables z2

        mean_pz1, scales_pz1 = self.VAE.p_z1(latent.to(self.VAE.device))
        p_z1 = Independent(Normal(mean_pz1, scales_pz1), 1)
        z1_gen = p_z1.rsample()

        reco_x = self.VAE.decoder(torch.cat((z1_gen, latent), 1)).cpu()
        decoded = reco_x.detach().transpose(1, 2)
        decoded = decoded.reshape((decoded.shape[0], -1))
        decoded = self.X.scaler.inverse_transform(decoded)

        return decoded

    def fit(self, X, **kwargs):
        return self

    def sample(self, n_samples: int):  # Tuple[ndarray[float], ndarray[float]]

        with torch.no_grad():

            # Display of latent variable z2 and the generation
            h2 = self.VAE.encoder_z2(self.X.data[:n_samples])
            q2 = self.VAE.lsr(h2)

            if self.sim_type == "generation":
                p_z2 = self.VAE.lsr.get_prior()
                z2 = p_z2.sample(torch.Size([n_samples])).squeeze(1)

                # z1_gen ~ p(z1|z2_gen)
                mean_pz1, scales_pz1 = self.VAE.p_z1(z2)
                p_z1 = Independent(Normal(mean_pz1, scales_pz1), 1)
                z1 = p_z1.rsample()

            if self.sim_type == "reconstruction":
                z2 = q2.rsample()
                h1_x = self.VAE.encoder_z1_x(self.X.data[:n_samples])
                h1_z2 = self.VAE.encoder_z1_z2(z2)
                h1 = torch.cat((h1_x, h1_z2), 1)
                h1 = self.VAE.encoder_z1_joint(h1)
                q1 = self.VAE.q_z1(h1)
                z1 = q1.rsample()

            gen_x = self.VAE.decoder(
                torch.cat((z1, z2), 1).to(self.VAE.device)
            ).cpu()

        gen_x = gen_x.detach().transpose(1, 2).reshape(gen_x.shape[0], -1)
        return gen_x, 0
