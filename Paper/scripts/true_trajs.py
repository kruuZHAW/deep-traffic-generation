# %%

from traffic.core import Traffic
import matplotlib.pyplot as plt
from traffic.core.projection import EuroPP
from traffic.drawing import countries
from traffic.data import navaids
from traffic.data import airports
import numpy as np

# %%
traffic = Traffic.from_file(
    "../../deep_traffic_generation/data/traffic_noga_tilFAF_train.pkl"
)

# %%

with plt.style.context("traffic"):
    fig, ax = plt.subplots(
        1, 1, figsize=(15, 15), subplot_kw=dict(projection=EuroPP()), dpi=300
    )
    # ax.add_feature(countries())

    # ax.set_extent((7.5, 9.5, 47, 48.5))

    traffic[:4000].plot(ax, alpha=0.1)

    # k = np.random.randint(len(traffic))
    k = 13017
    traffic[k].plot(ax, color="#1f77b4", lw=1.5)
    traffic[k].at_ratio(0.8).plot(
        ax,
        color="#1f77b4",
        zorder=3,
        s=600,
        shift=dict(units="dots", x=-60, y=60),
        text_kw=dict(
            fontname="Fira Sans",
            fontSize=18,
            ha="right",
            bbox=dict(
                boxstyle="round",
                edgecolor="none",
                facecolor="white",
                alpha=0.7,
                zorder=5,
            ),
        ),
    )

    airports["LSZH"].plot(ax, footprint=False, runways=dict(lw=1), labels=False)

    navaids["OSNEM"].plot(
        ax,
        zorder=5,
        marker="^",
        shift=dict(units="dots", x=45, y=-45),
        text_kw={"s": "FAP", "fontSize": 18, "va": "center"},
    )

    fig.savefig("true_trajs.png", transparent=False, dpi=300)

# %%
