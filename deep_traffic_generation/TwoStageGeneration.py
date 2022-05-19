from deep_traffic_generation.tcvae import TCVAE
from deep_traffic_generation.SecondStageVAE import VAE
from deep_traffic_generation.core.datasets import TrafficDataset

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
        first_stage_path: str,
        second_stage_path: str,
        dataset_params: TypedDict,
    ):
        filenames1 = next(
            walk(first_stage_path + "checkpoints/"), (None, None, [])
        )[2]
        filenames2 = next(
            walk(second_stage_path + "checkpoints/"), (None, None, [])
        )[2]

        self.first_stage = TCVAE.load_from_checkpoint(
            first_stage_path + "checkpoints/" + filenames1[0],
            hparams_file=first_stage_path + "/hparams.yaml",
            dataset_params=dataset_params,
        )
        self.first_stage.eval()

        self.second_stage = VAE.load_from_checkpoint(
            second_stage_path + "checkpoints/" + filenames2[0],
            hparams_file=second_stage_path + "/hparams.yaml",
        )
        self.second_stage.eval()

    def get_second_stage_scaler(self):

        h = self.first_stage.encoder(self.X.data)
        q = self.first_stage.lsr(h)
        z = q.rsample()
        input_SecondStage = z.detach().cpu()

        scaler2 = MinMaxScaler(feature_range=(-1, 1))
        scaler2.fit(input_SecondStage)
        self.scaler2 = scaler2

    def latent_spaces(
        self, n_samples: int
    ):  # Gives the latent spaces of the first stage, the second stage, and n_sample generated points within

        h = self.first_stage.encoder(self.X.data)
        q = self.first_stage.lsr(h)
        z_1 = q.rsample()
        z_2 = torch.Tensor(self.scaler2.transform(z_1.detach()))
        h_2 = self.second_stage.encoder(z_2)
        mu = self.second_stage.fc_mu(h_2)
        std = torch.exp(self.second_stage.fc_var(h_2) / 2)
        q_2 = torch.distributions.Normal(mu, std)
        u = q_2.rsample()

        p_u = torch.distributions.Normal(
            torch.zeros(self.second_stage.fc_mu.out_features),
            torch.ones(self.second_stage.fc_var.out_features),
        )
        u_gen = p_u.rsample((n_samples,))

        z_gen = self.second_stage.decoder(
            u_gen.to(self.second_stage.device)
        ).cpu()
        # z_gen = z_gen + torch.exp(self.second_stage.log_scale)*p_u.rsample((n_samples,))
        z_gen = torch.Tensor(self.scaler2.inverse_transform(z_gen.detach()))

        u_embeddings = np.concatenate(
            (u.detach().numpy(), u_gen.detach().numpy()), axis=0
        )
        z_embeddings = np.concatenate(
            (z_1.detach().numpy(), z_gen.detach().numpy()), axis=0
        )

        return u_embeddings, z_embeddings

    def decode(self, latent):  # decode some given latent variables

        # p_u = torch.distributions.Normal(
        #             torch.zeros(self.second_stage.fc_mu.out_features),
        #             torch.ones(self.second_stage.fc_var.out_features),
        #         )

        reco_z = self.second_stage.decoder(
            latent.to(self.second_stage.device)
        ).cpu()
        # reco_z = reco_z + torch.exp(SecondStage.log_scale)*p_u.rsample()
        reco_z = torch.Tensor(self.scaler2.inverse_transform(reco_z.detach()))
        reco_x = self.first_stage.decoder(reco_z)
        decoded = reco_x.detach().transpose(1, 2)
        decoded = decoded.reshape((decoded.shape[0], -1))
        decoded = self.X.scaler.inverse_transform(decoded)

        return decoded

    def fit(self, X, **kwargs):
        return self

    def sample(self, n_samples: int):  # Tuple[ndarray[float], ndarray[float]]

        with torch.no_grad():

            if self.sim_type == "generation":
                p_u = torch.distributions.Normal(
                    torch.zeros(self.second_stage.fc_mu.out_features),
                    torch.ones(self.second_stage.fc_var.out_features),
                )
                u = p_u.rsample((n_samples,))

            # The reconstruction process goes through the 2 stages, even though the only first stage is way more efficient
            # The reconstruction through the 1st stage only will be implemented into the OneStageVAE GenerationProtocol
            if self.sim_type == "reconstruction":
                h = self.first_stage.encoder(self.X.data[:n_samples])
                q = self.first_stage.lsr(h)
                z_1 = q.rsample()
                z_1 = z_1.detach().cpu().numpy()
                z_2 = torch.Tensor(self.scaler2.transform(z_1))
                h_2 = self.second_stage.encoder(z_2)
                mu = self.second_stage.fc_mu(h_2)
                std = torch.exp(self.second_stage.fc_var(h_2) / 2)
                q_2 = torch.distributions.Normal(mu, std)
                u = q_2.rsample()

            gen_z = self.second_stage.decoder(
                u.to(self.second_stage.device)
            ).cpu()
            # gen_z = gen_z + torch.exp(self.second_stage.log_scale)*p_u.rsample((n_samples,))
            gen_z = torch.Tensor(self.scaler2.inverse_transform(gen_z.detach()))
            gen_x = self.first_stage.decoder(gen_z)

        gen_x = gen_x.detach().transpose(1, 2).reshape(gen_x.shape[0], -1)
        return gen_x, 0
