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
grey = "#bab0ac"

with plt.style.context("traffic"):
    fig = plt.figure(figsize=(20, 30))
    ax0 = fig.add_subplot(421)
    ax1 = fig.add_subplot(422, projection=EuroPP())
    ax2 = fig.add_subplot(423)
    ax3 = fig.add_subplot(424, projection=EuroPP())
    ax4 = fig.add_subplot(425)
    ax5 = fig.add_subplot(426, projection=EuroPP())
    ax6 = fig.add_subplot(427)
    ax7 = fig.add_subplot(428, projection=EuroPP())

    ax0.scatter(Z_fcae.X1, Z_fcae.X2, s=4, c=grey)
    ax0.scatter(
        Z_fcae.query("label == 0.0").X1,
        Z_fcae.query("label == 0.0").X2,
        s=4,
        c=color_cycle[0],
    )
    ax0.set_yticklabels([])
    ax0.set_xticklabels([])
    ax0.set_title("FCAE latent space", fontsize=30, pad=18)
    ax0.grid(False)
    ax1.set_title("FCAE reconstructed trajectories", fontsize=30, pad=18)
    traffics_fcae[0].plot(ax1, alpha=0.2, color=color_cycle[0])

    ax2.scatter(Z_fcae.X1, Z_fcae.X2, s=4, c=grey)
    ax2.scatter(
        Z_fcae.query("label == 1.0").X1,
        Z_fcae.query("label == 1.0").X2,
        s=4,
        c=color_cycle[1],
    )
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.grid(False)
    traffics_fcae[1].plot(ax3, alpha=0.2, color=color_cycle[1])

    ax4.scatter(Z_fcae.X1, Z_fcae.X2, s=4, c=grey)
    ax4.scatter(
        Z_fcae.query("label == 2.0").X1,
        Z_fcae.query("label == 2.0").X2,
        s=4,
        c=color_cycle[2],
    )
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    ax4.grid(False)
    traffics_fcae[2].plot(ax5, alpha=0.2, color=color_cycle[2])

    ax6.scatter(Z_fcae.X1, Z_fcae.X2, s=4, c=grey)
    ax6.scatter(
        Z_fcae.query("label == 3.0").X1,
        Z_fcae.query("label == 3.0").X2,
        s=4,
        c=color_cycle[3],
    )
    ax6.set_yticklabels([])
    ax6.set_xticklabels([])
    ax6.grid(False)
    traffics_fcae[3].plot(ax7, alpha=0.2, color=color_cycle[3])

    plt.subplots_adjust(
        left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0.2
    )

fig.savefig("clustering_FCAE.png", transparent=False, dpi=300)

# %%
from traffic.core.projection import EuroPP

color_cycle = "#4c78a8 #f58518 #54a24b #f2cf5b".split()
colors_tcae = [color_cycle[int(i)] for i in Z_tcae.label]
grey = "#bab0ac"

with plt.style.context("traffic"):
    fig = plt.figure(figsize=(20, 30))
    ax0 = fig.add_subplot(421)
    ax1 = fig.add_subplot(422, projection=EuroPP())
    ax2 = fig.add_subplot(423)
    ax3 = fig.add_subplot(424, projection=EuroPP())
    ax4 = fig.add_subplot(425)
    ax5 = fig.add_subplot(426, projection=EuroPP())
    ax6 = fig.add_subplot(427)
    ax7 = fig.add_subplot(428, projection=EuroPP())

    ax0.scatter(Z_tcae.X1, Z_tcae.X2, s=4, c=grey)
    ax0.scatter(
        Z_tcae.query("label == 0.0").X1,
        Z_tcae.query("label == 0.0").X2,
        s=4,
        c=color_cycle[0],
    )
    ax0.set_yticklabels([])
    ax0.set_xticklabels([])
    ax0.set_title("TCAE latent space", fontsize=30, pad=18)
    ax0.grid(False)
    ax1.set_title("TCAE reconstructed trajectories", fontsize=30, pad=18)
    traffics_tcae[0].plot(ax1, alpha=0.2, color=color_cycle[0])

    ax2.scatter(Z_tcae.X1, Z_tcae.X2, s=4, c=grey)
    ax2.scatter(
        Z_tcae.query("label == 1.0").X1,
        Z_tcae.query("label == 1.0").X2,
        s=4,
        c=color_cycle[1],
    )
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.grid(False)
    traffics_tcae[1].plot(ax3, alpha=0.2, color=color_cycle[1])

    ax4.scatter(Z_tcae.X1, Z_tcae.X2, s=4, c=grey)
    ax4.scatter(
        Z_tcae.query("label == 2.0").X1,
        Z_tcae.query("label == 2.0").X2,
        s=4,
        c=color_cycle[2],
    )
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    ax4.grid(False)
    traffics_tcae[2].plot(ax5, alpha=0.2, color=color_cycle[2])

    ax6.scatter(Z_tcae.X1, Z_tcae.X2, s=4, c=grey)
    ax6.scatter(
        Z_tcae.query("label == 3.0").X1,
        Z_tcae.query("label == 3.0").X2,
        s=4,
        c=color_cycle[3],
    )
    ax6.set_yticklabels([])
    ax6.set_xticklabels([])
    ax6.grid(False)
    traffics_tcae[3].plot(ax7, alpha=0.2, color=color_cycle[3])

    plt.subplots_adjust(
        left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0.2
    )

fig.savefig("clustering_TCAE.png", transparent=False, dpi=300)
# %%
