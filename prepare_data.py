# prepare_data.py

import os
import glob
import pandas as pd
from fetch_f1_data import get_lap_data, get_pitstop_data


def compute_track_overtake_index(laps_folder: str, pit_folder: str, sc_events: pd.DataFrame | None = None) -> pd.Series:
    """Return average overtakes per lap for each track using historical data.

    Parameters
    ----------
    laps_folder : str
        Directory containing ``*_laps.csv`` files.
    pit_folder : str
        Directory containing ``*_pitstops.csv`` files.
    sc_events : pd.DataFrame, optional
        Optional DataFrame with columns ``raceId``, ``lap`` and ``sc_flag`` to
        mark safety-car laps that should be excluded.

    Returns
    -------
    pd.Series
        Series indexed by ``track_id`` with the average overtakes per driver-lap.
    """

    lap_files = glob.glob(os.path.join(laps_folder, "*_laps.csv"))
    frames: list[pd.DataFrame] = []
    for lap_path in lap_files:
        laps = pd.read_csv(lap_path)
        if laps.empty:
            continue

        # determine matching pitstop file
        base = os.path.basename(lap_path).replace("_laps.csv", "")
        pit_path = os.path.join(pit_folder, f"{base}_pitstops.csv")
        pits = pd.read_csv(pit_path) if os.path.exists(pit_path) else pd.DataFrame(columns=["driverId", "lap"])

        # mark pit in- and out-laps for exclusion
        outlaps = pits[["driverId", "lap"]].copy()
        if not outlaps.empty:
            outlaps["lap"] = outlaps["lap"] + 1
        exclude = pd.concat([pits[["driverId", "lap"]], outlaps], ignore_index=True)
        laps = laps.merge(exclude, on=["driverId", "lap"], how="left", indicator=True)
        laps = laps.query("_merge=='left_only'").drop(columns=["_merge"])

        # optionally remove safety car laps
        if sc_events is not None and not sc_events.empty:
            race_id = laps.get("raceId")
            if race_id is not None:
                race_id = race_id.iloc[0]
                sc = sc_events[(sc_events["raceId"] == race_id) & (sc_events["sc_flag"] == 1)]
                if not sc.empty:
                    laps = laps.merge(sc[["lap"]], on="lap", how="left", indicator=True)
                    laps = laps.query("_merge=='left_only'").drop(columns=["_merge"])

        laps = laps.sort_values(["driverId", "lap"])
        laps["prev_pos"] = laps.groupby("driverId")["position"].shift(1)
        laps["delta"] = (laps["prev_pos"] - laps["position"]).clip(lower=0).fillna(0)
        frames.append(laps[["track_id", "delta"]])

    if not frames:
        return pd.Series(dtype=float)

    all_laps = pd.concat(frames, ignore_index=True)
    return all_laps.groupby("track_id")["delta"].mean()




# --------------------------- Overtake calculation ---------------------------
def compute_overtakes(df_laps: pd.DataFrame, df_pits: pd.DataFrame, sc_events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calculate detailed overtake metrics per driver for a single race.

    Parameters
    ----------
    df_laps : pd.DataFrame
        Raw lap data for a single race containing ``driverId``, ``lap``,
        ``position``, ``raceId`` and ``track_id``.
    df_pits : pd.DataFrame
        Pit stop information for the same race to exclude pit in/out laps.
    sc_events : pd.DataFrame, optional
        Safety-car flags with columns ``raceId``, ``lap`` and ``sc_flag``.

    Returns
    -------
    pd.DataFrame
        Aggregated overtake metrics with columns ``driverId``,
        ``overtakes_count``, ``overtakes_rate``, ``overtakes_last10`` and
        ``overtakes_post_SC``. An extra ``valid_lap_count`` column is returned
        for debugging and rate calculation. The resulting DataFrame also
        contains ``track_overtake_index`` describing how easy overtaking
        historically is on the circuit.
    """

    if df_laps is None or df_laps.empty:
        return pd.DataFrame(
            columns=[
                "driverId",
                "overtakes_count",
                "overtakes_rate",
                "overtakes_last10",
                "overtakes_post_SC",
                "overtakes_lap1",
                "valid_lap_count",
                "track_overtake_index",
            ]
        )

    df = df_laps.sort_values(["driverId", "lap"]).copy()

    # map dynamic track index for later merge
    track_index = compute_track_overtake_index("data", "data", sc_events)
    df["track_overtake_index"] = df["track_id"].map(track_index).fillna(0)

    # prepare pit in/out lap exclusion
    if df_pits is None:
        df_pits = pd.DataFrame(columns=["driverId", "lap"])
    outlaps = df_pits[["driverId", "lap"]].copy()
    if not outlaps.empty:
        outlaps["lap"] = outlaps["lap"] + 1
    exclude = pd.concat([df_pits[["driverId", "lap"]], outlaps], ignore_index=True)
    df = df.merge(exclude, on=["driverId", "lap"], how="left", indicator=True)
    df = df.query("_merge=='left_only'").drop(columns=["_merge"])

    # fill missing safety-car column with zeros so normal races keep working
    if "sc_flag" not in df.columns:
        df["sc_flag"] = 0

    # previous position per driver to calculate position changes
    df["prev_pos"] = df.groupby("driverId")["position"].shift(1)
    df["delta"] = (df["prev_pos"] - df["position"]).clip(lower=0)

    # mark laps right after a safety car phase
    df["prev_sc"] = df.groupby("driverId")["sc_flag"].shift(1).fillna(0)
    df["post_sc"] = (df["prev_sc"] == 1) & (df["sc_flag"] == 0) & (df["delta"] > 0)

    # exclude laps where the safety car is out
    df_valid = df[df["sc_flag"] == 0]

    # total overtakes and valid lap count per driver
    agg = df_valid.groupby("driverId").agg(
        overtakes_count=("delta", "sum"),
        valid_lap_count=("lap", "count"),
    ).reset_index()

    # rate normalised by available laps
    agg["overtakes_rate"] = agg["overtakes_count"] / agg["valid_lap_count"].where(
        agg["valid_lap_count"] > 0, 1
    )

    # overtakes in the last 10 laps of the race
    last_lap = df["lap"].max()
    last10 = df_valid[df_valid["lap"] >= last_lap - 9]
    last10_counts = (
        last10.groupby("driverId")["delta"].sum().rename("overtakes_last10")
    )
    agg = agg.merge(last10_counts, on="driverId", how="left")

    # first lap overtakes (useful for start analysis)
    lap1_counts = (
        df_valid[df_valid["lap"] == 1]
        .groupby("driverId")["delta"].sum()
        .rename("overtakes_lap1")
    )
    agg = agg.merge(lap1_counts, on="driverId", how="left")

    # overtakes immediately after a safety car period
    post_sc_counts = (
        df_valid[df_valid["post_sc"]]
        .groupby("driverId")["delta"].sum()
        .rename("overtakes_post_SC")
    )
    agg = agg.merge(post_sc_counts, on="driverId", how="left")

    # fill missing values with zeros for clarity
    agg = agg.fillna(0)

    # every driver in a race shares the same track index -> take first
    agg["track_overtake_index"] = df["track_overtake_index"].iloc[0]

    return agg[
        [
            "driverId",
            "overtakes_count",
            "overtakes_rate",
            "overtakes_last10",
            "overtakes_post_SC",
            "overtakes_lap1",
            "valid_lap_count",
            "track_overtake_index",
        ]
    ]


def main():
    # 1. Bestandspaden
    files = {
        'qual':     'jolpica_qualifying.csv',
        'races':    'jolpica_races.csv',
        'results':  'jolpica_results.csv',
        'circuits': 'jolpica_circuits.csv',
        'sessions': 'openf1_sessions.csv',
        'weather':  'openf1_weather.csv'
    }

    # 2. Inladen CSV's
    df_qual     = pd.read_csv(files['qual'])
    df_races    = pd.read_csv(files['races'])
    df_results  = pd.read_csv(files['results'])
    df_circ     = pd.read_csv(files['circuits'])
    df_sessions = pd.read_csv(files['sessions'], parse_dates=['date_start'])
    df_weather  = pd.read_csv(files['weather'])

    # Preload lap and pit stop data with caching
    unique_races = df_races[['season', 'round']].drop_duplicates()
    for season, rnd in unique_races.itertuples(index=False):
        get_lap_data(season, rnd, use_cache=True)
        get_pitstop_data(season, rnd, use_cache=True)

    # calculate track level overtake index once from all cached data
    track_index = compute_track_overtake_index("data", "data")

    # Gemiddelde weersdata per sessie berekenen (ook gemiddelde regen)
    weather_agg = (
        df_weather.groupby('session_key')[
            ['air_temperature', 'track_temperature', 'rainfall']
        ]
        .mean()
        .reset_index()
    )

    # 3. Hernoemen kolommen
    df_qual     = df_qual.rename(columns={'position':'grid_position'})
    df_results  = df_results.rename(columns={
        'position':'finish_position',
        'Constructor.constructorId':'constructorId'
    })

    # 4. Merge kwalificatie + races metadata
    df = df_qual.merge(
        df_races[['season','round','raceName','date','Circuit.circuitId']],
        on=['season','round','raceName'], how='left'
    )

    # 5. Merge race-resultaten inclusief constructorId
    df = df.merge(
        df_results[['season','round','raceName','Driver.driverId','finish_position','constructorId']],
        on=['season','round','raceName','Driver.driverId'], how='left'
    )

    # --- Overtakes per race -------------------------------------------------
    over_frames = []
    for season, rnd in df_results[['season', 'round']].drop_duplicates().itertuples(index=False):
        laps = get_lap_data(season=season, round=rnd)
        pits = get_pitstop_data(season=season, round=rnd)
        if laps is None or laps.empty:
            continue
        if pits is None:
            pits = pd.DataFrame(columns=["driverId", "lap"])

        overtakes = compute_overtakes(laps, pits)
        overtakes['season'] = season
        overtakes['round'] = rnd
        over_frames.append(overtakes)

    if over_frames:
        df_overtakes = pd.concat(over_frames, ignore_index=True)
    else:
        df_overtakes = pd.DataFrame(
            columns=[
                'driverId',
                'overtakes_count',
                'overtakes_rate',
                'overtakes_last10',
                'overtakes_post_SC',
                'overtakes_lap1',
                'valid_lap_count',
                'track_overtake_index',
                'season',
                'round',
            ]
        )

    df = df.merge(
        df_overtakes.rename(columns={'driverId': 'Driver.driverId'}),
        on=['season', 'round', 'Driver.driverId'],
        how='left'
    )
    for col in [
        'overtakes_count',
        'overtakes_rate',
        'overtakes_last10',
        'overtakes_post_SC',
        'overtakes_lap1',
        'valid_lap_count',
        'track_overtake_index'
    ]:
        df[col] = df[col].fillna(0)

    # 6. Doelvariabele
    df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
    df['top3']            = df['finish_position'] <= 3

    # 7. Q-tijden naar seconden
    def to_seconds(t):
        m, s = t.split(':')
        return int(m)*60 + float(s)

    for col in ['Q1','Q2','Q3']:
        df[f'{col}_sec'] = pd.to_numeric(
            df[col].dropna().apply(to_seconds), errors='coerce'
        )

    # 8. Datum invoeren
    df['date']    = pd.to_datetime(df['date'])
    df['month']   = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday

    # 9. Impute kwalificatietijden per circuit
    for sec in ['Q1_sec','Q2_sec','Q3_sec']:
        med = df.groupby('Circuit.circuitId')[sec].transform('median')
        df[sec] = df[sec].fillna(med).fillna(df[sec].median())

    # 10. Circuit-features
    df = df.merge(
        df_circ[['circuitId','circuitName','Location.lat','Location.long','Location.locality','Location.country']],
        left_on='Circuit.circuitId', right_on='circuitId', how='left'
    ).rename(columns={
        'Location.lat':'circuit_lat', 'Location.long':'circuit_long',
        'Location.locality':'circuit_city', 'Location.country':'circuit_country'
    })

    # 11. Rolling averages per driver
    df = df.sort_values(['Driver.driverId','date'])
    df['avg_finish_pos'] = df.groupby('Driver.driverId')['finish_position'] \
                             .transform(lambda x: x.shift().expanding().mean())
    df['avg_grid_pos']   = df.groupby('Driver.driverId')['grid_position']   \
                             .transform(lambda x: x.shift().expanding().mean())
    df[['avg_finish_pos','avg_grid_pos']] = df[['avg_finish_pos','avg_grid_pos']].fillna(
        df[['avg_finish_pos','avg_grid_pos']].median())

    # 12. Rolling averages per constructor
    const_df = df_results.merge(
        df_races[['season','round','raceName']], on=['season','round','raceName'], how='left'
    )
    const_df = const_df.rename(columns={'Constructor.constructorId':'constructorId'})
    const_df = const_df.groupby(['constructorId','season','round'])['finish_position'].mean().reset_index()
    const_df = const_df.sort_values(['constructorId','season','round'])
    const_df['avg_const_finish'] = const_df.groupby('constructorId')['finish_position'] \
                                       .transform(lambda x: x.shift().expanding().mean())
    const_df['avg_const_finish'] = const_df['avg_const_finish'].fillna(const_df['avg_const_finish'].median())
    df = df.merge(
        const_df[['constructorId','season','round','avg_const_finish']],
        on=['season','round','constructorId'], how='left'
    )

    # 13. Qualifying-session mapping voor weerdata
    df_sessions['date_only'] = df_sessions['date_start'].dt.date
    qual_sessions = df_sessions[df_sessions['session_type']=='Qualifying'][['date_only','session_key']]
    df['date_only'] = df['date'].dt.date
    df = df.merge(qual_sessions, on='date_only', how='left')

    # 14. Weersdata mergen via session_key (geaggregeerd)
    df = df.merge(
        weather_agg,
        on='session_key', how='left'
    )

    # Mogelijke duplicaten verwijderen na het mergen van weerdata
    df = df.drop_duplicates(subset=['season','round','Driver.driverId'])

    # … na de existing rolling averages & weather-imputatie …

    # 14a. Grid-difference feature
    df['grid_diff'] = df['avg_grid_pos'] - df['grid_position']

    # 14b. Q3-difference feature (eerst per driver avg Q3_sec berekenen)
    driver_q3 = df.groupby('Driver.driverId')['Q3_sec'].transform('mean')
    df['Q3_diff'] = driver_q3 - df['Q3_sec']

    # 14c. Interaction grid × track temperature
    df['grid_temp_int'] = df['grid_position'] * df['track_temperature']

    # 14d. Track-specific overtaking difficulty index
    df['track_overtake_index'] = df['Circuit.circuitId'].map(track_index).fillna(0)

    # 14e. Weather features and interaction with overtakes
    df['weather_flag'] = (df['rainfall'] > 0).astype(int)
    df['overtakes_temp_int'] = df['overtakes_count'] * df['track_temperature']


    # 15. Impute weather
    for col in ['air_temperature', 'track_temperature', 'rainfall']:
        df[col] = df[col].fillna(df[col].median())

    # Drop helper cols
    df.drop(columns=['date_only','session_key'], inplace=True)

    # 16. Wegschrijven
    df.to_csv('processed_data.csv', index=False)
    print(f"processed_data.csv saved — {len(df)} rows, {df['top3'].sum()} top3 labels")

if __name__ == '__main__':
    main()
