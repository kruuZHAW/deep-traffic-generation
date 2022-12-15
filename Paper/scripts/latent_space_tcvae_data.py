# %%
from deep_traffic_generation.VAE_Generation import SingleStageVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDataset

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import pandas as pd
import pickle
from os import walk


# %%
# dataset_TCVAE = TrafficDataset.from_file(
#     "../../deep_traffic_generation/data/traffic_noga_tilFAF_train.pkl",
#     features=["track", "groundspeed", "altitude", "timedelta"],
#     scaler=MinMaxScaler(feature_range=(-1, 1)),
#     shape="image",
#     info_params={"features": ["latitude", "longitude"], "index": -1},
# )

# path = "../../deep_traffic_generation/lightning_logs/tcvae/version_1/"

# t_TCVAE = SingleStageVAE(X=dataset_TCVAE, sim_type="generation")
# t_TCVAE.load(path, dataset_TCVAE.parameters)
# g_TCVAE = Generation(
#     generation=t_TCVAE,
#     features=t_TCVAE.VAE.hparams.features,
#     scaler=dataset_TCVAE.scaler,
# )

dataset_FCVAE = TrafficDataset.from_file(
    "../../deep_traffic_generation/data/traffic_noga_tilFAF_train.pkl",
    features=["track", "groundspeed", "altitude", "timedelta"],
    scaler=MinMaxScaler(feature_range=(-1, 1)),
    shape="linear",
    info_params={"features": ["latitude", "longitude"], "index": -1},
)

path = "../../deep_traffic_generation/lightning_logs/fcvae/version_0/"

t_FCVAE = SingleStageVAE(X=dataset_FCVAE, sim_type="generation")
t_FCVAE.load(path, dataset_FCVAE.parameters)
g_FCVAE = Generation(
    generation=t_FCVAE,
    features=t_FCVAE.VAE.hparams.features,
    scaler=dataset_FCVAE.scaler,
)


# %%
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

n_gen = 1
# Z = t_TCVAE.latent_space(n_gen)
Z = t_FCVAE.latent_space(n_gen)

# Only fitted on train data
pca = PCA(n_components=2).fit(Z[:-n_gen])
Z_embedded = pca.transform(Z)
print("Explained variance ratio : ", pca.explained_variance_ratio_)

labels = GaussianMixture(n_components=4, random_state=0).fit_predict(Z_embedded)
print("Clustering done")

Z_embedded = np.append(Z_embedded, np.expand_dims(labels, axis=1), axis=1)
Z_embedded = pd.DataFrame(Z_embedded, columns=["X1", "X2", "label"])

traffics = []
for i in np.unique(labels):
    print("traffic : ", i)
    # decoded = t_TCVAE.decode(torch.Tensor(Z[labels == i]))
    decoded = t_FCVAE.decode(torch.Tensor(Z[labels == i]))
    # traf_clust = g_TCVAE.build_traffic(
    traf_clust = g_FCVAE.build_traffic(
        decoded,
        coordinates=dict(latitude=47.546585, longitude=8.447731),
        forward=False,
    )
    traf_clust = traf_clust.assign(cluster=lambda x: i)
    traffics.append(traf_clust)

# Z_embedded.to_pickle("Z_clust_tcvae.pkl")
Z_embedded.to_pickle("Z_clust_fcvae.pkl")

# with open("traffics_clust_tcvae.pkl", "wb") as f:
with open("traffics_clust_fcvae.pkl", "wb") as f:
    pickle.dump(traffics, f)

# %%
from sklearn.decomposition import PCA

n_gen = 2000
# Z = t_TCVAE.latent_space(n_gen)
Z = t_FCVAE.latent_space(n_gen)

# Only fitted on train data
pca = PCA(n_components=2).fit(Z[:-n_gen])
Z_embedded = pca.transform(Z)

z_type = np.array(["observed" for i in range(Z_embedded.shape[0])])
z_type[-n_gen:] = "generated"

Z_embedded = np.append(Z_embedded, np.expand_dims(z_type, axis=1), axis=1)
Z_embedded = pd.DataFrame(Z_embedded, columns=["X1", "X2", "type"])

# Z_embedded.to_pickle("Z_gen_tcvae.pkl")
Z_embedded.to_pickle("Z_gen_fcvae.pkl")

# %%
