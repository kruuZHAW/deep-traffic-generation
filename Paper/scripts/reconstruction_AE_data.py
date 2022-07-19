# %%
from deep_traffic_generation.VAE_Generation import SingleStageVAE
from deep_traffic_generation.AE_Generation import AutoEncoder
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDataset

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import numpy as np
from os import walk

# %%
dataset_FCAE = TrafficDataset.from_file(
    "../../deep_traffic_generation/data/traffic_noga_tilFAF_train.pkl",
    features=["track", "groundspeed", "altitude", "timedelta"],
    scaler=MinMaxScaler(feature_range=(-1, 1)),
    shape="linear",
    info_params={"features": ["latitude", "longitude"], "index": -1},
)

dataset_TCAE = TrafficDataset.from_file(
    "../../deep_traffic_generation/data/traffic_noga_tilFAF_train.pkl",
    features=["track", "groundspeed", "altitude", "timedelta"],
    scaler=MinMaxScaler(feature_range=(-1, 1)),
    shape="image",
    info_params={"features": ["latitude", "longitude"], "index": -1},
)

# %%
path = "../../deep_traffic_generation/lightning_logs/fcae/version_3/"
t_FCAE = AutoEncoder(X=dataset_FCAE, AE_type="FCAE")
t_FCAE.load(path, dataset_FCAE.parameters)
g_FCAE = Generation(
    generation=t_FCAE,
    features=t_FCAE.AE.hparams.features,
    scaler=dataset_FCAE.scaler,
)

path = "../../deep_traffic_generation/lightning_logs/tcae/version_9/"
t_TCAE = AutoEncoder(X=dataset_TCAE, AE_type="TCAE")
t_TCAE.load(path, dataset_TCAE.parameters)
g_TCAE = Generation(
    generation=t_TCAE,
    features=t_TCAE.AE.hparams.features,
    scaler=dataset_TCAE.scaler,
)

# %%
from traffic.core import Traffic

traffic = Traffic.from_file(
    "../../deep_traffic_generation/data/traffic_noga_tilFAF_train.pkl"
)

# %%

# Comparison of a trajectory and it's reconstructed counterpart for fcae
j = 10795

original = dataset_FCAE.data[j].unsqueeze(0)
if len(original.shape) >= 3:
    original = original.transpose(1, 2).reshape((original.shape[0], -1))
original = dataset_FCAE.scaler.inverse_transform(original)
original_traf = g_FCAE.build_traffic(
    original,
    coordinates=dict(latitude=47.546585, longitude=8.447731),
    forward=False,
)

reconstructed = t_FCAE.decode(
    t_FCAE.AE.encoder(dataset_FCAE.data[j].unsqueeze(0))
)
reconstructed_traf = g_FCAE.build_traffic(
    reconstructed,
    coordinates=dict(latitude=47.546585, longitude=8.447731),
    forward=False,
)

reconstruction_traf_fcae = traffic[j] + reconstructed_traf
reconstruction_traf_fcae.to_pickle("reconstruction_fcae.pkl")

# %%
# Comparison of a trajectory and it's reconstructed counterpart for tcae
j = 10795

original = dataset_TCAE.data[j].unsqueeze(0)
if len(original.shape) >= 3:
    original = original.transpose(1, 2).reshape((original.shape[0], -1))
original = dataset_TCAE.scaler.inverse_transform(original)
original_traf = g_TCAE.build_traffic(
    original,
    coordinates=dict(latitude=47.546585, longitude=8.447731),
    forward=False,
)

reconstructed = t_TCAE.decode(
    t_TCAE.AE.encoder(dataset_TCAE.data[j].unsqueeze(0))
)
reconstructed_traf = g_TCAE.build_traffic(
    reconstructed,
    coordinates=dict(latitude=47.546585, longitude=8.447731),
    forward=False,
)

reconstruction_traf_tcae = traffic[j] + reconstructed_traf
reconstruction_traf_tcae.to_pickle("reconstruction_tcae.pkl")


# %%
