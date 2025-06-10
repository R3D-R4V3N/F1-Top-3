# prepare_data.py

import pandas as pd
import ast
from fetch_f1_data import get_lap_data, get_pitstop_data

# In prepare_data.py, vervang je bestaande compute_overtakes door deze volledige, geüpdatete versie:

def compute_overtakes(valid_laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gewogen en gefilterde overtakes-per-driver voor één race:
      1) zet lap-tijd om naar seconden
      2) filter onrealistische (zeer trage) laps (>1.2× mediane lap-tijd)
      3) bereken netto plaatswinst per lap (prev_pos - pos, ≥0)
      4) weeg elke inhaalactie met (mediane_lap_tijd / lap_time_sec)
      5) agregeer per driver: onge­wogen + gewogen overtakes
      6) normaliseer per geldig aantal laps
    Retourneert DataFrame met kolommen:
      - driverId
      - overtakes_count
      - weighted_overtakes
      - overtakes_per_lap
      - weighted_overtakes_per_lap
    """
    if valid_laps_df.empty:
        return pd.DataFrame(columns=[
            "driverId",
            "overtakes_count",
            "weighted_overtakes",
            "overtakes_per_lap",
            "weighted_overtakes_per_lap"
        ])

    df = valid_laps_df.copy()

    # 1) Lap-tijd omzetten naar seconden
    def to_sec(t: str) -> float:
        mins, secs = t.split(':')
        return int(mins) * 60 + float(secs)
    df['lap_time_sec'] = df['time'].apply(to_sec)

    # 2) Filter onrealistische laps (zoals safety car of pitstop)
    med = df['lap_time_sec'].median()
    max_valid = 1.2 * med
    df = df[df['lap_time_sec'] <= max_valid]

    # 3) Netto plaatswinst per lap
    df = df.sort_values(['driverId', 'lap'])
    df['prev_pos'] = df.groupby('driverId')['position'].shift(1)
    df['pos_gain'] = (df['prev_pos'] - df['position']).clip(lower=0)

    # 4) Gewicht per lap gebaseerd op relatieve snelheid
    df['weight'] = med / df['lap_time_sec']
    df['weighted_gain'] = df['pos_gain'] * df['weight']

    # 5) Agregeer per driver
    agg = df.groupby('driverId').agg(
        overtakes_count     = ('pos_gain', 'sum'),
        weighted_overtakes  = ('weighted_gain', 'sum'),
        laps_count          = ('lap', 'count')
    ).reset_index()

    # 6) Normaliseer per geldig aantal laps
    agg['overtakes_per_lap']          = agg['overtakes_count'] / agg['laps_count']
    agg['weighted_overtakes_per_lap'] = agg['weighted_overtakes'] / agg['laps_count']

    return agg[[
        'driverId',
        'overtakes_count',
        'weighted_overtakes',
        'overtakes_per_lap',
        'weighted_overtakes_per_lap'
    ]]


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
    # Boolean indicator whether a driver finished the race
    df_results['did_finish'] = (
        df_results['status'].str.contains('Finished', case=False) |
        df_results['status'].str.match(r'\+\d+ Laps?') |
        (df_results['status'] == 'Lapped')
    )
    df_circ     = pd.read_csv(files['circuits'])
    df_sessions = pd.read_csv(files['sessions'], parse_dates=['date_start'])
    df_weather  = pd.read_csv(files['weather'])
    df_drvstand = pd.read_csv('jolpica_driverstandings.csv')
    df_constand = pd.read_csv('jolpica_constructorstandings.csv')

    # Preload lap and pit stop data with caching
    unique_races = df_races[['season', 'round']].drop_duplicates()
    for season, rnd in unique_races.itertuples(index=False):
        get_lap_data(season, rnd, use_cache=True)
        get_pitstop_data(season, rnd, use_cache=True)

    # (Weather features removed)

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
        df_results[['season','round','raceName','Driver.driverId',
                    'finish_position','constructorId','did_finish']],
        on=['season','round','raceName','Driver.driverId'],
        how='left'
    )

    # (Driver/constructor standings removed)

    # --- Overtakes per race -------------------------------------------------
    over_frames = []
    for season, rnd in df_results[['season', 'round']].drop_duplicates().itertuples(index=False):
        laps = get_lap_data(season=season, round=rnd)
        pits = get_pitstop_data(season=season, round=rnd)
        if laps is None or laps.empty:
            continue
        if pits is None:
            pits = pd.DataFrame(columns=["driverId", "lap"])

        valid_laps = laps.merge(
            pits[['driverId', 'lap']],
            on=['driverId', 'lap'],
            how='left',
            indicator=True
        )
        valid_laps = valid_laps.query("_merge=='left_only'").drop(columns=['_merge'])
        overtakes = compute_overtakes(valid_laps)
        overtakes['season'] = season
        overtakes['round'] = rnd
        over_frames.append(overtakes)

    if over_frames:
        df_overtakes = pd.concat(over_frames, ignore_index=True)
    else:
        df_overtakes = pd.DataFrame(columns=['driverId', 'overtakes_count', 'season', 'round'])

    df = df.merge(
        df_overtakes.rename(columns={'driverId': 'Driver.driverId'}),
        on=['season', 'round', 'Driver.driverId'],
        how='left'
    )

    # --- Pit stop stats & tyre degradation ---------------------------------
    pit_frames = []
    deg_frames = []
    for s, r in df_results[['season', 'round']].drop_duplicates().itertuples(index=False):
        pits = get_pitstop_data(season=s, round=r)
        laps = get_lap_data(season=s, round=r)
        if pits is not None and not pits.empty:
            def _pstop_to_sec(val: str) -> float:
                if pd.isna(val):
                    return None
                try:
                    if ':' in val:
                        m, sec = val.split(':')
                        return int(m) * 60 + float(sec)
                    return float(val)
                except Exception:
                    return None
            pits['duration_sec'] = pits['stopTime'].apply(_pstop_to_sec)
            agg = pits.groupby('driverId').agg(
                num_pitstops=('stop', 'count'),
                avg_pitstop_duration=('duration_sec', 'mean')
            ).reset_index()
            agg['season'] = s
            agg['round'] = r
            pit_frames.append(agg)

        if laps is not None and not laps.empty:
            def _lap_to_sec(t: str) -> float:
                mins, secs = t.split(':')
                return int(mins) * 60 + float(secs)
            laps['lap_sec'] = laps['time'].apply(_lap_to_sec)
            deg = laps.groupby('driverId')['lap_sec'].agg(['mean', 'min']).reset_index()
            deg['tyre_degradation_rate'] = deg['mean'] - deg['min']
            deg = deg[['driverId', 'tyre_degradation_rate']]
            deg['season'] = s
            deg['round'] = r
            deg_frames.append(deg)

    df_pits = pd.concat(pit_frames, ignore_index=True) if pit_frames else pd.DataFrame(
        columns=['driverId', 'num_pitstops', 'avg_pitstop_duration', 'season', 'round']
    )
    df_deg = pd.concat(deg_frames, ignore_index=True) if deg_frames else pd.DataFrame(
        columns=['driverId', 'tyre_degradation_rate', 'season', 'round']
    )

    df = df.merge(
        df_pits.rename(columns={'driverId': 'Driver.driverId'}),
        on=['season', 'round', 'Driver.driverId'],
        how='left'
    )
    df = df.merge(
        df_deg.rename(columns={'driverId': 'Driver.driverId'}),
        on=['season', 'round', 'Driver.driverId'],
        how='left'
    )

    df = df.sort_values(['Driver.driverId', 'date'])
    for col in ['num_pitstops', 'avg_pitstop_duration', 'tyre_degradation_rate']:
        df[col] = df.groupby('Driver.driverId')[col].shift().fillna(0)

    # Use only past races for overtake features to avoid leakage
    df = df.sort_values(['Driver.driverId', 'date'])
    df['overtakes_count'] = (
        df.groupby('Driver.driverId')['overtakes_count']
          .shift()
    )
    df['weighted_overtakes'] = (
        df.groupby('Driver.driverId')['weighted_overtakes']
          .shift()
    )
    df['overtakes_per_lap'] = (
        df.groupby('Driver.driverId')['overtakes_per_lap']
          .shift()
    )
    df['weighted_overtakes_per_lap'] = (
        df.groupby('Driver.driverId')['weighted_overtakes_per_lap']
          .shift()
    )
    df['overtakes_count'] = df['overtakes_count'].fillna(0)
    df['weighted_overtakes'] = df['weighted_overtakes'].fillna(0)
    df['overtakes_per_lap'] = df['overtakes_per_lap'].fillna(0)
    df['weighted_overtakes_per_lap'] = df['weighted_overtakes_per_lap'].fillna(0)

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

    # Qualifying delta: difference between final position and Q2 rank
    q2_ranks = (
        df.dropna(subset=['Q2_sec'])
          .sort_values(['season', 'round', 'Q2_sec'])
          .groupby(['season', 'round'])
          .cumcount() + 1
    )
    df.loc[~df['Q2_sec'].isna(), 'Q2_rank'] = q2_ranks.values
    df['qual_delta'] = df['grid_position'] - df['Q2_rank']
    df['qual_delta'] = df['qual_delta'].fillna(0)

    # 8. Datum invoeren
    df['date']    = pd.to_datetime(df['date'])
    df['month']   = df['date'].dt.month
    df['Driver.dateOfBirth'] = pd.to_datetime(df['Driver.dateOfBirth'])

    # 9. Impute kwalificatietijden per circuit
    # Sorteer op datum zodat we circuit-medians alleen uit voorgaande races
    # berekenen om datalek te vermijden. Per circuit wordt een lopende mediaan
    # bijgehouden op basis van eerdere waarden. De huidige race telt dus niet
    # mee in die mediaan.
    df = df.sort_values('date')
    for sec in ['Q1_sec', 'Q2_sec', 'Q3_sec']:
        running_med = (
            df.groupby('Circuit.circuitId')[sec]
              .transform(lambda s: s.expanding().median().shift())
        )
        global_med = df[sec].expanding().median().shift()
        df[sec] = df[sec].fillna(running_med)
        df[sec] = df[sec].fillna(global_med)
        df[sec] = df[sec].fillna(0)

    # --- Team qualifying gap -----------------------------------------------
    df['qual_best_sec'] = df[['Q1_sec', 'Q2_sec', 'Q3_sec']].min(axis=1, skipna=True)
    team_best = df.groupby(['season', 'round', 'constructorId'])['qual_best_sec'].transform('min')
    df['team_qual_gap'] = (df['qual_best_sec'] - team_best).fillna(0)
    df.drop(columns=['qual_best_sec'], inplace=True)

    # 10. Circuit-features
    df = df.merge(
        df_circ[['circuitId','circuitName','Location.lat','Location.long','Location.locality','Location.country']],
        left_on='Circuit.circuitId', right_on='circuitId', how='left'
    ).rename(columns={
        'Location.lat':'circuit_lat', 'Location.long':'circuit_long',
        'Location.locality':'circuit_city', 'Location.country':'circuit_country'
    })

    # Rolling finish rate over previous 5 races per driver
    df = df.sort_values(['Driver.driverId', 'date'])
    df['finish_rate_prev5'] = (
        df.groupby('Driver.driverId')['did_finish']
          .transform(lambda s: s.shift().rolling(window=5, min_periods=1).mean())
    )
    df['finish_rate_prev5'] = df['finish_rate_prev5'].fillna(
        df['finish_rate_prev5'].median()
    )

    # 11. Rolling averages per driver
    df = df.sort_values(['Driver.driverId','date'])
    df['avg_finish_pos'] = df.groupby('Driver.driverId')['finish_position'] \
                             .transform(lambda x: x.shift().expanding().mean())
    df['avg_grid_pos']   = df.groupby('Driver.driverId')['grid_position']   \
                             .transform(lambda x: x.shift().expanding().mean())
    df = df.sort_values('date')
    for col in ['avg_finish_pos', 'avg_grid_pos']:
        run_med = df[col].expanding().median().shift()
        df[col] = df[col].fillna(run_med)
        df[col] = df[col].fillna(0)

    # 12. Rolling averages per constructor
    const_df = df_results.merge(
        df_races[['season','round','raceName']], on=['season','round','raceName'], how='left'
    )
    const_df = const_df.rename(columns={'Constructor.constructorId':'constructorId'})
    const_df = const_df.groupby(['constructorId','season','round'])['finish_position'].mean().reset_index()
    const_df = const_df.sort_values(['constructorId','season','round'])
    const_df['avg_const_finish'] = const_df.groupby('constructorId')['finish_position'] \
                                       .transform(lambda x: x.shift().expanding().mean())
    const_df = const_df.sort_values(['season', 'round'])
    run_med = const_df['avg_const_finish'].expanding().median().shift()
    const_df['avg_const_finish'] = const_df['avg_const_finish'].fillna(run_med)
    const_df['avg_const_finish'] = const_df['avg_const_finish'].fillna(0)
    df = df.merge(
        const_df[['constructorId','season','round','avg_const_finish']],
        on=['season','round','constructorId'], how='left'
    )

    # Circuit top-3 frequency based on historical results
    res_full = df_results.merge(
        df_races[['season','round','date','Circuit.circuitId']],
        on=['season','round','raceName'],
        how='left'
    )
    res_full['finish_position'] = pd.to_numeric(res_full['finish_position'], errors='coerce')
    res_full['top3'] = res_full['finish_position'] <= 3
    res_full = res_full.sort_values('date')
    res_full['cum_top3'] = res_full.groupby('Circuit.circuitId')['top3'].cumsum().shift()
    res_full['cum_total'] = res_full.groupby('Circuit.circuitId').cumcount()
    res_full['circuit_top3_freq'] = res_full['cum_top3'] / res_full['cum_total'].replace(0, pd.NA)
    circ_freq = res_full[['season','round','Circuit.circuitId','circuit_top3_freq']].drop_duplicates(
        subset=['season','round','Circuit.circuitId']
    )
    df = df.merge(
        circ_freq,
        left_on=['season','round','Circuit.circuitId'],
        right_on=['season','round','Circuit.circuitId'],
        how='left'
    )
    df['circuit_top3_freq'] = df['circuit_top3_freq'].fillna(0)

    # Head-to-head vs teammate (running win rate)
    team_res = res_full[['season','round','date','Driver.driverId','constructorId','finish_position']].dropna(subset=['finish_position'])
    team_res = team_res.sort_values(['season','round','constructorId','finish_position'])
    team_res['rank_within_team'] = team_res.groupby(['season','round','constructorId']).cumcount() + 1
    team_res['beat_teammate'] = team_res['rank_within_team'] == 1
    hh = team_res[['season','round','Driver.driverId','date','beat_teammate']]
    hh = hh.sort_values(['Driver.driverId','date'])
    hh['head_to_head_vs_teammate'] = hh.groupby('Driver.driverId')['beat_teammate'].transform(lambda s: s.shift().expanding().mean())
    df = df.merge(
        hh[['season','round','Driver.driverId','head_to_head_vs_teammate']],
        on=['season','round','Driver.driverId'],
        how='left'
    )
    df['head_to_head_vs_teammate'] = df['head_to_head_vs_teammate'].fillna(0)

    # (Weather merge removed)

    # … na de existing rolling averages & weather-imputatie …

    # 14a. Grid-difference feature
    df['grid_diff'] = df['avg_grid_pos'] - df['grid_position']

    # 14b. Q3-difference feature op basis van voorgaande races
    df = df.sort_values(['Driver.driverId', 'date'])
    driver_q3 = df.groupby('Driver.driverId')['Q3_sec'].transform(
        lambda s: s.shift().expanding().mean()
    )
    df['Q3_diff'] = driver_q3 - df['Q3_sec']

    # 14c. Interaction grid × track temperature (removed)


    # 15. Drop helper cols
    df.drop(columns=['date_only','session_key'], errors='ignore', inplace=True)

    # Na:
    df['overtakes_per_lap']         = df['overtakes_per_lap'].fillna(0)
    df['weighted_overtakes_per_lap'] = df['weighted_overtakes_per_lap'].fillna(0)

    # Voeg toe: EWMA over de vorige 3 races (span=3) per driver
    df = df.sort_values(['Driver.driverId', 'date'])
    df['ewma_overtakes_per_lap'] = (
        df.groupby('Driver.driverId')['overtakes_per_lap']
        .transform(lambda s: s.ewm(span=3, adjust=False).mean())
    ).fillna(0)

    df['ewma_weighted_overtakes_per_lap'] = (
        df.groupby('Driver.driverId')['weighted_overtakes_per_lap']
        .transform(lambda s: s.ewm(span=3, adjust=False).mean())
    ).fillna(0)

    drop_cols = [
        'air_temperature', 'track_temperature', 'humidity', 'pressure',
        'rainfall', 'wind_speed', 'wind_direction',
        'constructor_points_prev', 'constructor_rank_prev', 'driver_age',
        'grid_temp_int', 'driver_points_prev', 'driver_rank_prev',
        'weekday', 'overtakes_count'
    ]
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # 16. Wegschrijven
    df.to_csv('processed_data.csv', index=False)
    print(f"processed_data.csv saved — {len(df)} rows, {df['top3'].sum()} top3 labels")

if __name__ == '__main__':
    main()
