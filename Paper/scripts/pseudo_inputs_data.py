# %%
from deep_traffic_generation.VAE_Generation import SingleStageVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDataset

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import pandas as pd
from os import walk

# %%
dataset_TCVAE = TrafficDataset.from_file(
    "../../deep_traffic_generation/data/traffic_noga_tilFAF_train.pkl",
    features=["track", "groundspeed", "altitude", "timedelta"],
    scaler=MinMaxScaler(feature_range=(-1, 1)),
    shape="image",
    info_params={"features": ["latitude", "longitude"], "index": -1},
)

path = "../../deep_traffic_generation/lightning_logs/tcvae/version_1/"

t_TCVAE = SingleStageVAE(X=dataset_TCVAE, sim_type="generation")
t_TCVAE.load(path, dataset_TCVAE.parameters)
g_TCVAE = Generation(
    generation=t_TCVAE,
    features=t_TCVAE.VAE.hparams.features,
    scaler=dataset_TCVAE.scaler,
)

# %%
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

# Computing the pseudo inputs

# Vampprior
pseudo_X = t_TCVAE.VAE.lsr.pseudo_inputs_NN(t_TCVAE.VAE.lsr.idle_input)
pseudo_X = pseudo_X.view((pseudo_X.shape[0], 4, 200))
pseudo_h = t_TCVAE.VAE.encoder(pseudo_X)
pseudo_means = t_TCVAE.VAE.lsr.z_loc(pseudo_h)
pseudo_scales = (t_TCVAE.VAE.lsr.z_log_var(pseudo_h) / 2).exp()

# Reconstructed pseudo-inputs
out = t_TCVAE.decode(pseudo_means)
# Neural net don't predict exaclty timedelta = 0 for the first observation
out[:, 3] = 0
# The track prediction is filtered (smoothen trajectories) : introduce small loops
# out[:, 0::4] = savgol_filter(out[:, 0::4], 11, 3)
out_traf = g_TCVAE.build_traffic(
    out, coordinates=dict(latitude=47.546585, longitude=8.447731), forward=False
)

# latent spaces from train dataset and pseudo_inputs
z_train = t_TCVAE.latent_space(1)
Z = np.concatenate((z_train, pseudo_means.detach().numpy()), axis=0)

pca = PCA(n_components=2).fit(Z[: -len(pseudo_means)])
Z_embedded = pca.transform(Z)
col = torch.norm(pseudo_scales, dim=1, p=2).detach().numpy()

Z_embedded = pd.DataFrame(Z_embedded, columns=["X1", "X2"])
Z_embedded["Scales"] = np.NaN
Z_embedded.Scales[-len(col) :] = col

Z_embedded.to_pickle("Z_pseudo_inputs.pkl")
out_traf.to_pickle("traffic_pseudo_inputs.pkl")

# %%
