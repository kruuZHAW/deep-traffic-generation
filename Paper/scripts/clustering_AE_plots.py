# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from os import walk

os.chdir("../../Paper/scripts")

# %%
with open("traffics_clust_fcae.pkl", "rb") as f:
    traffics_fcae = mynewlist = pickle.load(f)

with open("traffics_clust_tcae.pkl", "rb") as f:
    traffics_tcae = mynewlist = pickle.load(f)

Z_fcae = pd.read_pickle("Z_clust_fcae.pkl")
Z_tcae = pd.read_pickle("Z_clust_tcae.pkl")

# %%
from traffic.core.projection import EuroPP

color_cycle = "#4c78a8 #f58518 #54a24b #f2cf5b".split()
colors_fcae = [color_cycle[int(i)] for i in Z_fcae.label]

with plt.style.context("traffic"):
    fig = plt.figure(figsize=(30, 15))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122, projection=EuroPP())

    scat1 = ax0.scatter(Z_fcae.X1, Z_fcae.X2, s=4, c=colors_fcae)
    ax0.set_title("FCAE latent space", fontsize=46)

    ax1.set_title("FCAE reconstructed trajectories", fontsize=46)
    for i, traf in enumerate(traffics_fcae):
        traf.plot(ax1, alpha=0.2, color=color_cycle[i])

    plt.subplots_adjust(
        left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0.35
    )

fig.savefig("clustering_FCAE.png", transparent=False, dpi=300)

# %%
from traffic.core.projection import EuroPP

color_cycle = "#4c78a8 #f58518 #54a24b #f2cf5b".split()
colors_tcae = [color_cycle[int(i)] for i in Z_tcae.label]

with plt.style.context("traffic"):
    fig = plt.figure(figsize=(30, 15))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122, projection=EuroPP())

    scat1 = ax0.scatter(Z_tcae.X1, Z_tcae.X2, s=4, c=colors_tcae)
    ax0.set_title("TCAE latent space", fontsize=46)

    ax1.set_title("TCAE reconstructed trajectories", fontsize=46)
    for i, traf in enumerate(traffics_tcae):
        traf.plot(ax1, alpha=0.2, color=color_cycle[i])

    plt.subplots_adjust(
        left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0.35
    )

fig.savefig("clustering_TCAE.png", transparent=False, dpi=300)
# %%
