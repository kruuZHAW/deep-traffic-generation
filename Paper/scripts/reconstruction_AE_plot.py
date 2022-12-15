# %%
from traffic.core import Traffic
from traffic.core.projection import EuroPP, PlateCarree
from traffic.drawing import countries
import matplotlib.pyplot as plt

# %%
reconstruction_fcae = Traffic.from_file("reconstruction_fcae.pkl")
reconstruction_tcae = Traffic.from_file("reconstruction_tcae.pkl")

# reconstruction_fcae = Traffic.from_file("reconstruction_fcvae.pkl")
# reconstruction_tcae = Traffic.from_file("reconstruction_tcvae.pkl")

# %%
with plt.style.context("traffic"):
    fig, ax = plt.subplots(
        1, 2, figsize=(13, 8), subplot_kw=dict(projection=EuroPP())
    )

    ax[0].set_title("FCVAE reconstruction", pad=20, fontsize=20)
    reconstruction_fcae[0].plot(ax[0], lw=2)
    reconstruction_fcae[1].plot(ax[0], lw=2)

    ax[1].set_title("TCVAE reconstruction", pad=20, fontsize=20)
    reconstruction_tcae[0].plot(ax[1], lw=2, label="original")
    reconstruction_tcae[1].plot(ax[1], lw=2, label="reconstructed")
    ax[1].set_extent(ax[0].get_extent(crs=PlateCarree()))
    legend = fig.legend(
        loc="lower center", bbox_to_anchor=(0.5, 0.2), ncol=2, fontsize=18
    )
    legend.get_frame().set_edgecolor("none")

    fig.savefig("reconstruction_AE.png", transparent=False, dpi=300)

# %%
