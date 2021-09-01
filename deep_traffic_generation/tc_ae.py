# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from deep_traffic_generation.core import AE, TCN
from deep_traffic_generation.core.datasets import TrafficDataset
from deep_traffic_generation.core.protocols import TransformerProtocol
from deep_traffic_generation.core.utils import cli_main


# fmt: on
class LinearAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class TCEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        h_dims: List[int],
        seq_len: int,
        kernel_size: int,
        dilation_base: int,
        sampling_factor: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            TCN(
                input_dim,
                h_dims[-1],
                h_dims[:-1],
                kernel_size,
                dilation_base,
                h_activ,
                dropout,
            ),
            nn.AvgPool1d(sampling_factor),
            nn.Flatten(),
            nn.Linear(h_dims[-1] * (int(seq_len / sampling_factor)), out_dim)
            # We might want to add a non-linear activation
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


class TCDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        h_dims: List[int],
        seq_len: int,
        kernel_size: int,
        dilation_base: int,
        sampling_factor: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.sampling_factor = sampling_factor

        self.decode_entry = nn.Linear(
            input_dim, h_dims[0] * int(seq_len / sampling_factor)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=sampling_factor),
            TCN(
                h_dims[0],
                out_dim,
                h_dims[1:],
                kernel_size,
                dilation_base,
                h_activ,
                dropout,
            ),
        )

    def forward(self, x):
        x = self.decode_entry(x)
        b, _ = x.size()
        x = x.view(b, -1, int(self.seq_len / self.sampling_factor))
        x_hat = self.decoder(x)
        return x_hat


class TCAE(AE):
    """Temporal Convolutional Autoencoder

    Source: http://www.gm.fh-koeln.de/ciopwebpub/Thill20a.d/bioma2020-tcn.pdf
    """

    _required_hparams = AE._required_hparams + [
        "sampling_factor",
        "kernel_size",
        "dilation_base",
    ]

    def __init__(
        self,
        x_dim: int,
        seq_len: int,
        scaler: Optional[TransformerProtocol],
        navpts: Optional[torch.Tensor],
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(x_dim, seq_len, scaler, navpts, config)

        self.example_input_array = torch.rand(
            (self.input_dim, self.seq_len)
        ).unsqueeze(0)

        self.encoder = TCEncoder(
            input_dim=x_dim,
            out_dim=self.hparams.encoding_dim,
            h_dims=self.hparams.h_dims,
            seq_len=self.seq_len,
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            # h_activ=nn.ReLU(),
            dropout=self.hparams.dropout,
        )

        self.decoder = TCDecoder(
            input_dim=self.hparams.encoding_dim,
            out_dim=x_dim,
            h_dims=self.hparams.h_dims[::-1],
            seq_len=self.seq_len,
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            # h_activ=nn.ReLU(),
            dropout=self.hparams.dropout,
        )

        # non-linear activations
        self.out_activ = LinearAct()  # nn.Tanh()

    # Training with Soft Dynamic Time Warping
    # def training_step(self, batch, batch_idx):
    #     x, _, _ = batch
    #     z = self.encoder(x)
    #     x_hat = self.out_activ(self.decoder(z))
    #     x_T = torch.transpose(x, 1, 2)
    #     x_hat_T = torch.transpose(x_hat, 1, 2)
    #     loss = sdtw_loss(x_T, x_hat_T)
    #     self.log("train_loss", loss)
    #     return loss

    def test_step(self, batch, batch_idx):
        x, _, info = batch
        z = self.encoder(x)
        x_hat = self.out_activ(self.decoder(z))
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return torch.transpose(x, 1, 2), torch.transpose(x_hat, 1, 2), info

    @classmethod
    def network_name(cls) -> str:
        return "tc_ae"

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--sampling_factor",
            dest="sampling_factor",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--kernel",
            dest="kernel_size",
            type=int,
            default=16,
        )
        parser.add_argument(
            "--dilation",
            dest="dilation_base",
            type=int,
            default=2,
        )

        return parent_parser, parser


if __name__ == "__main__":
    cli_main(TCAE, TrafficDataset, "image", seed=42)
