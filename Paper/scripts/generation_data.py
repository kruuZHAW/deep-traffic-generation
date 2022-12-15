# %%
from deep_traffic_generation.VAE_Generation import SingleStageVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
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
pseudo_X = t_TCVAE.VAE.lsr.pseudo_inputs_NN(t_TCVAE.VAE.lsr.idle_input)
pseudo_X = pseudo_X.view((pseudo_X.shape[0], 4, 200))
pseudo_h = t_TCVAE.VAE.encoder(pseudo_X)
pseudo_means = t_TCVAE.VAE.lsr.z_loc(pseudo_h)
pseudo_scales = (t_TCVAE.VAE.lsr.z_log_var(pseudo_h) / 2).exp()

# # %%
# for k in range(len(pseudo_means)):
#     test = t_TCVAE.decode(pseudo_means[k].unsqueeze(0))
#     test = g_TCVAE.build_traffic(test, coordinates = dict(latitude = 47.546585, longitude = 8.447731), forward=False)
#     if test[0].shape.is_simple == False:
#         print(k)

# # %%
# k = 96
# test = t_TCVAE.decode(pseudo_means[k].unsqueeze(0))
# test = g_TCVAE.build_traffic(test, coordinates = dict(latitude = 47.546585, longitude = 8.447731), forward=False)
# test[0]

# %%
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

# js = [262, 481]
# js = [262, 787]
js = [39, 91]
n_gen = 100

dist1 = torch.distributions.Independent(
    torch.distributions.Normal(pseudo_means[js[0]], pseudo_scales[js[0]]), 1
)
gen1 = dist1.sample(torch.Size([n_gen]))

dist2 = torch.distributions.Independent(
    torch.distributions.Normal(pseudo_means[js[1]], pseudo_scales[js[1]]), 1
)
gen2 = dist2.sample(torch.Size([n_gen]))

decode1 = t_TCVAE.decode(
    torch.cat((pseudo_means[js[0]].unsqueeze(0), gen1), axis=0)
)
decode2 = t_TCVAE.decode(
    torch.cat((pseudo_means[js[1]].unsqueeze(0), gen2), axis=0)
)

# Neural net don't predict exaclty timedelta = 0 for the first observation
decode1[:, 3] = 0
decode2[:, 3] = 0
# C'est le filtre qui introduit ces petites loops
# decode1[:, 0::4] = savgol_filter(decode1[:, 0::4], 11, 3)
# decode2[:, 0::4] = savgol_filter(decode2[:, 0::4], 11, 3)

traf_gen1 = g_TCVAE.build_traffic(
    decode1,
    coordinates=dict(latitude=47.546585, longitude=8.447731),
    forward=False,
)
traf_gen1 = traf_gen1.assign(gen_number=lambda x: 1)

traf_gen2 = g_TCVAE.build_traffic(
    decode2,
    coordinates=dict(latitude=47.546585, longitude=8.447731),
    forward=False,
)
traf_gen2 = traf_gen2.assign(gen_number=lambda x: 2)

z_train = t_TCVAE.latent_space(1)
gen = torch.cat((gen1, gen2, pseudo_means[js]), axis=0)
concat = np.concatenate((z_train, gen.detach().numpy()))
pca = PCA(n_components=2).fit(concat[: -len(gen)])
gen_embedded = pca.transform(concat)

gen_embedded = pd.DataFrame(gen_embedded, columns=["X1", "X2"])
gen_embedded["type"] = np.nan
gen_embedded.type[-(2 * n_gen + 2) :] = "GEN1"
gen_embedded.type[-(n_gen + 2) :] = "GEN2"
gen_embedded.type[-2:] = "PI1"
gen_embedded.type[-1:] = "PI2"

gen_embedded.to_pickle("Z_generated.pkl")
traf_gen1.to_pickle("traffic_generated_1.pkl")
traf_gen2.to_pickle("traffic_generated_2.pkl")
# %%
