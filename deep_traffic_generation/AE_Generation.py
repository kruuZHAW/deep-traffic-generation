from deep_traffic_generation.tcae import TCAE
from deep_traffic_generation.fcae import FCAE
from deep_traffic_generation.core.datasets import TrafficDataset

import torch
import numpy as np

from os import walk
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, TypedDict

""" Class adapted to use the Generation method for simple autoencoders
Sample and fit are not implemented 
"""


class AutoEncoder:
    def __init__(
        self,
        X: TrafficDataset,  # Traffic dataset used to train
        AE_type: str,
    ):
        super().__init__()

        self.X = X
        self.AE_type = AE_type

        if AE_type not in ["FCAE", "TCAE"]:
            raise ValueError(
                "Invalid sim type. Expected one of: %s" % ["FCAE", "TCAE"]
            )

    def load(
        self,
        path: str,
        dataset_params: TypedDict,
    ):
        filenames = next(walk(path + "checkpoints/"), (None, None, []))[2]

        if self.AE_type == "FCAE":
            self.AE = FCAE.load_from_checkpoint(
                path + "checkpoints/" + filenames[0],
                hparams_file=path + "/hparams.yaml",
                dataset_params=dataset_params,
            )
        if self.AE_type == "TCAE":
            self.AE = TCAE.load_from_checkpoint(
                path + "checkpoints/" + filenames[0],
                hparams_file=path + "/hparams.yaml",
                dataset_params=dataset_params,
            )
        self.AE.eval()

    def latent_space(self):
        """
        Return the latent space of the autoencoder
        """

        z = self.AE.encoder(self.X.data)

        return z.detach().numpy()

    def decode(self, latent):  # decode some given latent variables

        reco_x = self.AE.decoder(latent.to(self.AE.device)).cpu()
        if self.AE_type == "TCAE":
            decoded = reco_x.detach().transpose(1, 2)
            decoded = decoded.reshape((decoded.shape[0], -1))
        else:
            decoded = reco_x.detach().numpy()
        decoded = self.X.scaler.inverse_transform(decoded)

        return decoded

    def fit(self, X, **kwargs):
        return self

    def sample(self, n_samples: int):
        return self, 0
