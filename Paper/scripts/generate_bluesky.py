from typing import Iterator, Optional

import pandas as pd

from traffic.core import Flight, Traffic


def aligned_stats(traj: "Flight") -> Optional[pd.DataFrame]:
    from traffic.data import navaids

    navaids_extent = navaids.extent(traj, buffer=0.1)
    if navaids_extent is None:
        return None

    df = pd.DataFrame.from_records(
        list(
            {
                "start": segment.start,
                "stop": segment.stop,
                "duration": segment.duration,
                "navaid": segment.max("navaid"),
                "distance": segment.min("distance"),
                "bdiff_mean": segment.data[
                    "shift"
                ].mean(),  # b_diff_mean is equivalent to shift.mean() now ?
                "bdiff_meanp": segment.data["shift"].mean() + 0.02,
            }
            for segment in traj.aligned_on_navpoint(
                list(navaids_extent.drop_duplicates("name"))
            )
        )
    )

    if df.shape[0] == 0:
        return None
    return df.sort_values("start")


def groupby_intervals(table: pd.DataFrame) -> "Iterator[pd.DataFrame]":
    if table.shape[0] == 0:
        return
    table = table.sort_values("start")
    sweeping_line = table.query(
        "stop <= stop.iloc[0]"
    )  # take as much as you can
    # then try to push the stop line: which intervals overlap the stop line
    additional = table.query("start <= @sweeping_line.stop.max() < stop")

    while additional.shape[0] > 0:
        sweeping_line = table.query("stop <= @additional.stop.max()")
        additional = table.query("start <= @sweeping_line.stop.max() < stop")

    yield sweeping_line
    yield from groupby_intervals(
        table.query("start > @sweeping_line.stop.max()")
    )


def reconstruct_navpoints(traj: "Flight") -> "Iterator[pd.DataFrame]":
    table = aligned_stats(traj)
    if table is None:
        return
    for block in groupby_intervals(table):
        t_threshold = block.eval("duration.max()") - pd.Timedelta(  # noqa: F841
            "30s"
        )
        yield block.sort_values("bdiff_mean").query(
            "duration >= @t_threshold"
        ).head(1)


def navpoints_table(flight: "Flight") -> Optional["Flight"]:
    from traffic.data import navaids

    navaids_extent = navaids.extent(flight, buffer=0.1)
    if navaids_extent is None:
        return None

    list_ = list(reconstruct_navpoints(flight))
    if len(list_) == 0:
        print(f"fail with {flight.flight_id}")
        return None

    navpoints_table = pd.concat(list(reconstruct_navpoints(flight))).merge(
        navaids_extent.drop_duplicates("name").data,
        left_on="navaid",
        right_on="name",
    )

    cd_np = navaids_extent.drop_duplicates("name")

    try_list = list(
        (i, cd_np[elt.navaid], elt.stop, elt.duration)
        for i, elt in navpoints_table.assign(
            delta=lambda df: df.start.shift(-1) - df.stop
        )
        .drop(
            columns=[
                "altitude",
                "frequency",
                "magnetic_variation",
                "description",
            ]
        )
        .query('delta > "30s"')
        .iterrows()
    )
    for i, fix, stop, duration in try_list:
        cd = (
            flight.after(stop)
            .first(duration)  # type: ignore
            .assign(track=lambda df: df.track + 180)
            .aligned_on_navpoint([fix])
            .next()
        )
        if cd is not None:

            navpoints_table.loc[i, ["stop", "distance"]] = (
                cd.stop,
                -cd.distance_max,
            )

    return Flight(
        navpoints_table.assign(
            flight_id=flight.flight_id,
            callsign=flight.callsign,
            icao24=flight.icao24,
            registration=flight.registration,
            typecode=flight.typecode,
            runway=flight.runway_max,
            coverage=navpoints_table.duration.sum().total_seconds()
            / flight.duration.total_seconds(),
            latitude_0=flight.at_ratio(0).latitude,  # type: ignore
            longitude_0=flight.at_ratio(0).longitude,  # type: ignore
            altitude_0=flight.at_ratio(0).altitude,  # type: ignore
            track_0=flight.at_ratio(0).track,  # type: ignore
            groundspeed_0=flight.at_ratio(0).groundspeed,  # type: ignore
        ).drop(
            columns=[
                "bdiff_mean",
                "bdiff_meanp",
                "frequency",
                "magnetic_variation",
                "description",
                "altitude",
            ]
        )
    )


def main():

    gentrajs_tcvae = Traffic.from_file("../outputs/TCVAE_generation_5000.pkl")

    tcvae_bluesky = (
        gentrajs_tcvae.iterate_lazy()
        .assign(runway="14")
        .pipe(navpoints_table)
        .eval(desc="navpoints table", max_workers=40)
    )
    tcvae_bluesky.to_csv("../outputs/tcvae_generation_navpoints.csv")
    return

    from traffic.data.datasets import landing_zurich_2019

    navpoints = (
        landing_zurich_2019.query("simple")
        .iterate_lazy()
        .pipe(navpoints_table)
        .eval(desc="", max_workers=10)
    )

    # navpoints = list(
    #     navpoints_table(flight)
    #     for flight in tqdm(landing_zurich_2019.query("simple"))
    # )

    navpoints.to_csv("landing_zurich_2019_navpoints.csv")

    douglas_peucker = (
        landing_zurich_2019.query("simple")
        .iterate_lazy()
        .simplify(5e3)
        .eval(desc="", max_workers=10)
    )

    douglas_peucker.to_csv("landing_zurich_2019_douglas_peucker_5e3.csv")


if __name__ == "__main__":
    main()
