# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from os import walk

# %%
# with open("traffics_clust_tcvae.pkl", "rb") as f:
#     traffics_tcvae = mynewlist = pickle.load(f)

# Z_tcvae_clust = pd.read_pickle("Z_clust_tcvae.pkl")
# Z_tcvae_gen = pd.read_pickle("Z_gen_tcvae.pkl")

with open("traffics_clust_fcvae.pkl", "rb") as f:
    traffics_tcvae = mynewlist = pickle.load(f)

Z_tcvae_clust = pd.read_pickle("Z_clust_fcvae.pkl")
Z_tcvae_gen = pd.read_pickle("Z_gen_fcvae.pkl")

# %%
from traffic.core.projection import EuroPP

color_cycle = "#4c78a8 #f58518 #b79a20 #54a24b #439894 #e45756 #b279a2".split()
colors_tcvae = [color_cycle[int(i)] for i in Z_tcvae_clust.label]

with plt.style.context("traffic"):
    fig = plt.figure(figsize=(30, 15))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122, projection=EuroPP())

    scat1 = ax0.scatter(Z_tcvae_clust.X1, Z_tcvae_clust.X2, s=4, c=colors_tcvae)
    ax0.set_title("FCVAE latent space", fontsize=46)

    ax1.set_extent((7.5, 9.5, 46.8, 48.3))
    ax1.set_title("FCVAE reconstructed trajectories", fontsize=46)
    for i, traf in enumerate(traffics_tcvae):
        traf.plot(ax1, alpha=0.2, color=color_cycle[i])

    plt.subplots_adjust(
        left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0.35
    )

# fig.savefig("clustering_TCVAE.png", transparent=False, dpi=300)
fig.savefig("clustering_FCVAE.png", transparent=False, dpi=300)


# %%
observed = Z_tcvae_gen.query("type == 'observed'").astype(
    {"X1": "double", "X2": "double"}
)
generated = Z_tcvae_gen.query("type == 'generate'").astype(
    {"X1": "double", "X2": "double"}
)

with plt.style.context("traffic"):
    fig, ax = plt.subplots(1, figsize=(30, 15))

    ax.scatter(observed.X1, observed.X2, s=16, c="#bab0ac", label="Observed")
    ax.scatter(generated.X1, generated.X2, s=16, c="#4c78a8", label="Generated")
    ax.set_title("Latent Space", fontsize=46)

    legend = fig.legend(
        loc="upper right", bbox_to_anchor=(0.9, 0.85), fontsize=32
    )
    legend.get_frame().set_edgecolor("none")
    legend.legendHandles[0]._sizes = [100]
    legend.legendHandles[1]._sizes = [100]

# fig.savefig("generation_latent_space_TCVAE.png", transparent=False, dpi=300)

# %%
