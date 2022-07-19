# %%
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from traffic.core import Traffic
from os import walk

# %%

Z_pseudo_inputs = pd.read_pickle("Z_pseudo_inputs.pkl")
traffic_pseudo_inputs = Traffic.from_file("traffic_pseudo_inputs.pkl")

# %%
from traffic.core.projection import EuroPP
from traffic.data import airports

Z_observed = Z_pseudo_inputs.query("Scales.isnull()")
Z_pi = Z_pseudo_inputs.query("Scales.notna()").reset_index(drop=True)

with plt.style.context("traffic"):
    fig = plt.figure(figsize=(15, 12))
    ax0 = fig.add_subplot(221)
    ax1 = fig.add_subplot(222, projection=EuroPP())

    # k = np.random.randint(Z_pi.shape[0])
    k = 787

    ax0.scatter(
        Z_observed.X1, Z_observed.X2, c="#bab0ac", s=4, label="observed"
    )
    points = ax0.scatter(
        Z_pi.X1,
        Z_pi.X2,
        s=8,
        c=Z_pi.Scales,
        cmap="viridis",
        label="Pseudo-inputs",
    )
    ax0.scatter(
        Z_observed.X1[k],
        Z_observed.X2[k],
        s=50,
        marker="o",
        c="#f58518",
        label="Selected pseudo-input",
    )
    ax0.set_title("Latent Space", fontsize=18)

    legend = ax0.legend(loc="upper left", fontsize=12)
    legend.get_frame().set_edgecolor("none")
    legend.legendHandles[0]._sizes = [50]
    legend.legendHandles[1]._sizes = [50]
    legend.legendHandles[2]._sizes = [50]

    fig.colorbar(points, ax=ax0)

    ax1.set_extent((7.5, 9.5, 47, 48.5))
    ax1.set_title("Reconstructed synthetic pseudo-inputs", fontsize=18)
    traffic_pseudo_inputs.plot(ax1, alpha=0.2)
    traffic_pseudo_inputs["TRAJ_" + str(k)].plot(ax1, color="#f58518", lw=2)
    traffic_pseudo_inputs["TRAJ_" + str(k)].at_ratio(0.85).plot(
        ax1,
        color="#f58518",
        zorder=5,
        text_kw={"s": None},
    )

    airports["LSZH"].point.plot(ax1)
    fig.tight_layout()
    # plt.subplots_adjust(
    #     left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0
    # )

fig.savefig("pseudo_inputs.png", transparent=False, dpi=300)
# %%
