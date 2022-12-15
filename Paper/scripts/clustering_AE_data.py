# %%
from deep_traffic_generation.AE_Generation import AutoEncoder
from deep_traffic_generation.VAE_Generation import SingleStageVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDataset

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
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
# path = "../../deep_traffic_generation/lightning_logs/fcae/version_3/"
# t_FCAE = AutoEncoder(X=dataset_FCAE, AE_type="FCAE")
# t_FCAE.load(path, dataset_FCAE.parameters)
# g_FCAE = Generation(
#     generation=t_FCAE,
#     features=t_FCAE.AE.hparams.features,
#     scaler=dataset_FCAE.scaler,
# )

# path = "../../deep_traffic_generation/lightning_logs/tcae/version_9/"
# t_TCAE = AutoEncoder(X=dataset_TCAE, AE_type="TCAE")
# t_TCAE.load(path, dataset_TCAE.parameters)
# g_TCAE = Generation(
#     generation=t_TCAE,
#     features=t_TCAE.AE.hparams.features,
#     scaler=dataset_TCAE.scaler,
# )

path = "../../deep_traffic_generation/lightning_logs/fcvae/version_0/"
t_FCVAE = SingleStageVAE(X=dataset_FCAE, sim_type="generation")
t_FCVAE.load(path, dataset_FCAE.parameters)
g_FCVAE = Generation(
    generation=t_FCVAE,
    features=t_FCVAE.VAE.hparams.features,
    scaler=dataset_FCAE.scaler,
)

path = "../../deep_traffic_generation/lightning_logs/tcvae/version_1/"
t_TCVAE = SingleStageVAE(X=dataset_TCAE, sim_type="generation")
t_TCVAE.load(path, dataset_TCAE.parameters)
g_TCVAE = Generation(
    generation=t_TCVAE,
    features=t_TCVAE.VAE.hparams.features,
    scaler=dataset_TCAE.scaler,
)

# %%
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
import pickle
import pandas as pd
import numpy as np

# Z = t_FCAE.latent_space()
Z = t_FCVAE.latent_space(1)

pca = PCA(n_components=2).fit(Z)
Z_embedded = pca.transform(Z)
print("Explained variance ratio : ", pca.explained_variance_ratio_)

# Clustering on the pca transformation
# labels = SpectralClustering(n_clusters=4, gamma=8.0).fit_predict(Z_embedded)
labels = GaussianMixture(n_components=4, random_state=0).fit_predict(Z_embedded)
print("Clustering done")
Z_embedded = np.append(Z_embedded, np.expand_dims(labels, axis=1), axis=1)
Z_embedded = pd.DataFrame(Z_embedded, columns=["X1", "X2", "label"])

traffics = []
for i in np.unique(labels):
    print("traffic : ", i)
    # decoded = t_FCAE.decode(torch.Tensor(Z[labels == i]))
    decoded = t_FCVAE.decode(torch.Tensor(Z[labels == i]))
    # traf_clust = g_FCAE.build_traffic(
    traf_clust = g_FCVAE.build_traffic(
        decoded,
        coordinates=dict(latitude=47.546585, longitude=8.447731),
        forward=False,
    )
    traf_clust = traf_clust.assign(cluster=lambda x: i)
    traffics.append(traf_clust)

# Z_embedded.to_pickle("Z_clust_fcae.pkl")
Z_embedded.to_pickle("Z_clust_fcvae.pkl")

# with open("traffics_clust_fcae.pkl", "wb") as f:
#     pickle.dump(traffics, f)
with open("traffics_clust_fcvae.pkl", "wb") as f:
    pickle.dump(traffics, f)


# %%
# Z = t_TCAE.latent_space()
Z = t_TCVAE.latent_space(1)

pca = PCA(n_components=2).fit(Z)
Z_embedded = pca.transform(Z)
print("Explained variance ratio : ", pca.explained_variance_ratio_)

# Clustering on the pca transformation
# labels = SpectralClustering(n_clusters=4, gamma=8.0).fit_predict(Z_embedded)
labels = GaussianMixture(n_components=4, random_state=0).fit_predict(Z_embedded)
print("Clustering done")
Z_embedded = np.append(Z_embedded, np.expand_dims(labels, axis=1), axis=1)
Z_embedded = pd.DataFrame(Z_embedded, columns=["X1", "X2", "label"])

traffics = []
for i in np.unique(labels):
    print("traffic : ", i)
    # decoded = t_TCAE.decode(torch.Tensor(Z[labels == i]))
    decoded = t_TCVAE.decode(torch.Tensor(Z[labels == i]))
    # traf_clust = g_TCAE.build_traffic(
    traf_clust = g_TCVAE.build_traffic(
        decoded,
        coordinates=dict(latitude=47.546585, longitude=8.447731),
        forward=False,
    )
    traf_clust = traf_clust.assign(cluster=lambda x: i)
    traffics.append(traf_clust)

# Z_embedded.to_pickle("Z_clust_tcae.pkl")
Z_embedded.to_pickle("Z_clust_tcvae.pkl")

# with open("traffics_clust_tcae.pkl", "wb") as f:
#     pickle.dump(traffics, f)
with open("traffics_clust_tcvae.pkl", "wb") as f:
    pickle.dump(traffics, f)

# %%
