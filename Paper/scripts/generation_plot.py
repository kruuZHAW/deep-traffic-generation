# %%
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from traffic.core import Traffic
from os import walk

# %%
Z_gen = pd.read_pickle("Z_generated.pkl")
traf_gen_1 = Traffic.from_file("traffic_generated_1.pkl")
traf_gen_2 = Traffic.from_file("traffic_generated_2.pkl")

# %%
from traffic.core.projection import EuroPP
from traffic.data import airports

with plt.style.context("traffic"):
    fig = plt.figure(figsize=(15, 12))
    ax0 = fig.add_subplot(221)
    ax1 = fig.add_subplot(222, projection=EuroPP())

    ax0.scatter(
        Z_gen.query("type.isnull()").X1,
        Z_gen.query("type.isnull()").X2,
        c="#bab0ac",
        s=4,
        label="Observed",
    )
    ax0.scatter(
        Z_gen.query("type == 'GEN1'").X1,
        Z_gen.query("type == 'GEN1'").X2,
        c="#9ecae9",
        s=8,
        label="Generation pseudo_input 1",
    )
    ax0.scatter(
        Z_gen.query("type == 'GEN2'").X1,
        Z_gen.query("type == 'GEN2'").X2,
        c="#ffbf79",
        s=8,
        label="Generation pseudo-input 2",
    )
    ax0.scatter(
        Z_gen.query("type == 'PI1'").X1,
        Z_gen.query("type == 'PI1'").X2,
        c="#4c78a8",
        s=50,
        label="Pseudo-input 1",
    )
    ax0.scatter(
        Z_gen.query("type == 'PI2'").X1,
        Z_gen.query("type == 'PI2'").X2,
        c="#f58518",
        s=50,
        label="Pseudo-input 2",
    )
    ax0.set_title("Latent Space", fontsize=18)

    legend = ax0.legend(loc="upper left", fontsize=12)
    legend.get_frame().set_edgecolor("none")
    legend.legendHandles[0]._sizes = [50]
    legend.legendHandles[1]._sizes = [50]
    legend.legendHandles[2]._sizes = [50]

    ax1.set_title("Generated synthetic trajectories", pad=100, fontsize=18)

    traf_gen_1.plot(ax1, alpha=0.2, color="#9ecae9")
    traf_gen_1["TRAJ_0"].plot(ax1, color="#4c78a8", lw=2)
    traf_gen_1["TRAJ_0"].at_ratio(0.5).plot(
        ax1,
        color="#4c78a8",
        zorder=5,
        text_kw={"s": None},
    )

    traf_gen_2.plot(ax1, alpha=0.2, color="#ffbf79")
    traf_gen_2["TRAJ_0"].plot(ax1, color="#f58518", lw=2)
    traf_gen_2["TRAJ_0"].at_ratio(0.5).plot(
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

fig.savefig("generation.png", transparent=False, dpi=300)

# %%
import altair as alt

# Just put the pseudo-input at the end for display
copy_traf_1 = traf_gen_1
a = copy_traf_1["TRAJ_0"].assign(flight_id="TRAJ_999")
copy_traf_1 = copy_traf_1 + a

copy_traf_2 = traf_gen_2
b = copy_traf_2["TRAJ_0"].assign(flight_id="TRAJ_999")
copy_traf_2 = copy_traf_2 + b

chart1 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "timedelta",
                title="timedelta (in s)",
            ),
            y=alt.Y("altitude", title=None),
            opacity=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value(1),
                alt.value(0.2),
            ),
            color=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value("#4c78a8"),
                alt.value("#9ecae9"),
            ),
        )
        for flight in copy_traf_1
    )
).properties(title="altitude (in ft)")

chart2 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "timedelta",
                title="timedelta (in s)",
            ),
            y=alt.Y("groundspeed", title=None),
            opacity=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value(1),
                alt.value(0.2),
            ),
            color=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value("#4c78a8"),
                alt.value("#9ecae9"),
            ),
        )
        for flight in copy_traf_1
    )
).properties(title="groundspeed (in kts)")

chart3 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "timedelta",
                title="timedelta (in s)",
            ),
            y=alt.Y("altitude", title=None),
            opacity=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value(1),
                alt.value(0.2),
            ),
            color=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value("#f58518"),
                alt.value("#ffbf79"),
            ),
        )
        for flight in copy_traf_2
    )
).properties(title="altitude (in ft)")

chart4 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "timedelta",
                title="timedelta (in s)",
            ),
            y=alt.Y("groundspeed", title=None),
            opacity=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value(1),
                alt.value(0.2),
            ),
            color=alt.condition(
                alt.datum.flight_id == "TRAJ_999",
                alt.value("#f58518"),
                alt.value("#ffbf79"),
            ),
        )
        for flight in copy_traf_2
    )
).properties(title="groundspeed (in kts)")

plots = (
    alt.vconcat(alt.hconcat(chart1, chart2), alt.hconcat(chart3, chart4))
    .configure_title(fontSize=18)
    .configure_axis(labelFontSize=12, titleFontSize=14)
)

plots
# plots.save('alt_gs_gen.png', scale_factor=2.0)

# %%
