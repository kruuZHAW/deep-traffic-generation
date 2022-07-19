# %%
from traffic.core import Traffic

traffic = Traffic.from_file(
    "../../deep_traffic_generation/data/t_lszh_2019_04.pkl"
)
landing = (
    traffic.next('aligned_on_ils("LSZH")')
    .last("1 min")
    .eval(max_workers=50, desc="", cache_file="landing.pkl")
)
take_off = (
    traffic.next('takeoff_from_runway("LSZH")')
    .query("altitude < 10000")
    .eval(max_workers=50, desc="", cache_file="takeoff.pkl")
)

# %%

import matplotlib.pyplot as plt
from traffic.data import airports
from cartes.crs import EuroPP

airport = airports["LSZH"]

with plt.style.context("traffic"):
    fig, ax = plt.subplots(
        1, 3, figsize=(10, 7), subplot_kw=dict(projection=EuroPP()), dpi=300
    )

    for ax_ in ax:
        airport.plot(ax_, footprint=False, runways=dict(linewidth=4))
        ax_.spines["geo"].set_visible(False)
        ax_.set_extent(airport, buffer=0.05)

    ax[0].text(0.1, 0.1, "a. South concept", transform=ax[0].transAxes)
    ax[1].text(0.1, 0.1, "b. North concept", transform=ax[1].transAxes)
    ax[2].text(0.1, 0.1, "c. East concept", transform=ax[2].transAxes)

    (
        f := landing.query('ILS == "34"')[0]
        .distance(airport)
        .query("distance < 2")
    ).plot(ax=ax[0], linewidth=2, color="#4c78a8")
    f.at_ratio(0).plot(
        ax=ax[0], text_kw=dict(s="34"), shift=dict(units="dots", x=60), zorder=5
    )

    (
        f := take_off.query('runway == "28"')[0]
        .airborne()
        .distance(airport)
        .query("distance < 2")
    ).plot(ax=ax[0], linewidth=2, color="#4c78a8")
    f.at_ratio(1).plot(
        ax=ax[0], text_kw=dict(s="28"), shift=dict(units="dots", x=60), zorder=5
    )

    (
        f := take_off.query('runway == "32"')[1]
        .airborne()
        .distance(airport)
        .query("distance < 4")
    ).plot(ax=ax[0], linewidth=2, color="#4c78a8")
    f.at_ratio(1).plot(
        ax=ax[0], text_kw=dict(s="32"), shift=dict(units="dots", x=60), zorder=5
    )

    (
        f := landing.query('ILS == "14"')[0]
        .distance(airport)
        .query("distance < 4")
    ).plot(ax=ax[1], linewidth=2, color="#4c78a8")
    f.at_ratio(0).plot(
        ax=ax[1],
        shift=dict(units="dots", x=-60),
        text_kw=dict(s="14", ha="right"),
        zorder=5,
    )

    (
        f := landing.query('ILS == "16"')[3]
        .distance(airport)
        .query("distance < 4")
    ).plot(ax=ax[1], linewidth=2, color="#4c78a8")
    f.at_ratio(0).plot(
        ax=ax[1], text_kw=dict(s="16"), shift=dict(units="dots", x=60), zorder=5
    )

    (
        f := take_off.query('runway == "28"')[0]
        .airborne()
        .distance(airport)
        .query("distance < 2")
    ).plot(ax=ax[1], linewidth=2, color="#4c78a8")
    f.at_ratio(1).plot(
        ax=ax[1], text_kw=dict(s="28"), shift=dict(units="dots", x=60), zorder=5
    )

    (
        f := take_off.query('runway == "16"')[1]
        .airborne()
        .distance(airport)
        .query("distance < 2")
    ).plot(ax=ax[1], linewidth=2, color="#4c78a8")
    f.at_ratio(1).plot(
        ax=ax[1], text_kw=dict(s="16"), shift=dict(units="dots", x=60), zorder=5
    )

    (
        f := landing.query('ILS == "28"')[3]
        .distance(airport)
        .query("distance < 2")
    ).plot(ax=ax[2], linewidth=2, color="#4c78a8")
    f.at_ratio(0).plot(
        ax=ax[2], text_kw=dict(s="28"), shift=dict(units="dots", x=60), zorder=5
    )

    (
        f := take_off.query('runway == "28"')[0]
        .airborne()
        .distance(airport)
        .query("distance < 2")
    ).plot(ax=ax[2], linewidth=2, color="#4c78a8")
    f.at_ratio(1).plot(
        ax=ax[2], text_kw=dict(s="28"), shift=dict(units="dots", x=60), zorder=5
    )

    (
        f := take_off.query('runway == "32"')[1]
        .airborne()
        .distance(airport)
        .query("distance < 4")
    ).plot(ax=ax[2], linewidth=2, color="#4c78a8")
    f.at_ratio(1).plot(
        ax=ax[2], text_kw=dict(s="32"), shift=dict(units="dots", x=60), zorder=5
    )

    fig.savefig("concepts.png", transparent=False, dpi=300)
    # fig.set_tight_layout(True)


# %%
