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

    # Gemiddelde weersdata per sessie berekenen
    weather_agg = df_weather.groupby('session_key')[['air_temperature','track_temperature']].mean().reset_index()

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

    # --- Driver and constructor standings ---------------------------------
    driver_records = []
    for _, row in df_drvstand.iterrows():
        for entry in ast.literal_eval(row['DriverStandings']):
            driver_records.append({
                'season': row['season'],
                'round': row['round'],
                'Driver.driverId': entry['Driver']['driverId'],
                'driver_points': float(entry['points']),
                'driver_rank': int(entry['position'])
            })
    drv_df = pd.DataFrame(driver_records)
    drv_df = drv_df.sort_values(['Driver.driverId','season','round'])
    drv_df['driver_points_prev'] = (
        drv_df.groupby('Driver.driverId')['driver_points']
              .transform(lambda x: x.shift().expanding().mean())
    )
    drv_df['driver_rank_prev'] = (
        drv_df.groupby('Driver.driverId')['driver_rank']
              .transform(lambda x: x.shift().expanding().mean())
    )

    const_records = []
    for _, row in df_constand.iterrows():
        for entry in ast.literal_eval(row['ConstructorStandings']):
            const_records.append({
                'season': row['season'],
                'round': row['round'],
                'constructorId': entry['Constructor']['constructorId'],
                'constructor_points': float(entry['points']),
                'constructor_rank': int(entry['position'])
            })
    const_df = pd.DataFrame(const_records)
    const_df = const_df.sort_values(['constructorId','season','round'])
    const_df['constructor_points_prev'] = (
        const_df.groupby('constructorId')['constructor_points']
                .transform(lambda x: x.shift().expanding().mean())
    )
    const_df['constructor_rank_prev'] = (
        const_df.groupby('constructorId')['constructor_rank']
                .transform(lambda x: x.shift().expanding().mean())
    )

    df = df.merge(
        drv_df[['season','round','Driver.driverId','driver_points','driver_rank','driver_points_prev','driver_rank_prev']],
        on=['season','round','Driver.driverId'], how='left'
    )
    df = df.merge(
        const_df[['season','round','constructorId','constructor_points','constructor_rank','constructor_points_prev','constructor_rank_prev']],
        on=['season','round','constructorId'], how='left'
    )

    # Fill missing previous-standings with column medians
    prev_cols = [
        'driver_points_prev', 'driver_rank_prev',
        'constructor_points_prev', 'constructor_rank_prev'
    ]
    df[prev_cols] = df[prev_cols].fillna(df[prev_cols].median())

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
    df['overtakes_count'] = df['overtakes_count'].fillna(0)
    df['weighted_overtakes'] = df['weighted_overtakes'].fillna(0)

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
        df[sec] = df[sec].fillna(running_med)
        df[sec] = df[sec].fillna(df[sec].median())

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

    # Grid position relative to teammate (previous races only)
    def teammate_diff(s: pd.Series) -> pd.Series:
        if len(s) <= 1:
            return pd.Series(0, index=s.index)
        return s - (s.sum() - s) / (len(s) - 1)

    df['grid_vs_teammate'] = (
        df.groupby(['season', 'round', 'constructorId'])['grid_position']
          .transform(teammate_diff)
    )
    df['grid_vs_teammate'] = (
        df.sort_values(['Driver.driverId', 'date'])
          .groupby('Driver.driverId')['grid_vs_teammate']
          .shift()
    )
    df['grid_vs_teammate'] = df['grid_vs_teammate'].fillna(df['grid_vs_teammate'].median())

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

    # 14b. Q3-difference feature op basis van voorgaande races
    df = df.sort_values(['Driver.driverId', 'date'])
    driver_q3 = df.groupby('Driver.driverId')['Q3_sec'].transform(
        lambda s: s.shift().expanding().mean()
    )
    df['Q3_diff'] = driver_q3 - df['Q3_sec']

    # 14c. Interaction grid × track temperature
    df['grid_temp_int'] = df['grid_position'] * df['track_temperature']


    # 15. Impute weather
    for col in ['air_temperature','track_temperature']:
        df[col] = df[col].fillna(df[col].median())

    # Drop helper cols
    df.drop(columns=['date_only','session_key'], inplace=True)

    # Na:
    df['overtakes_per_lap']         = df['overtakes_per_lap'].fillna(0)
    df['weighted_overtakes_per_lap'] = df['weighted_overtakes_per_lap'].fillna(0)

    # Voeg toe: EWMA over de vorige 3 races (span=3) per driver
    df = df.sort_values(['Driver.driverId', 'date'])
    df['ewma_overtakes_per_lap'] = (
        df.groupby('Driver.driverId')['overtakes_per_lap']
        .transform(lambda s: s.shift().ewm(span=3, adjust=False).mean())
    ).fillna(0)

    df['ewma_weighted_overtakes_per_lap'] = (
        df.groupby('Driver.driverId')['weighted_overtakes_per_lap']
        .transform(lambda s: s.shift().ewm(span=3, adjust=False).mean())
    ).fillna(0)

    # 16. Wegschrijven
    df.to_csv('processed_data.csv', index=False)
    print(f"processed_data.csv saved — {len(df)} rows, {df['top3'].sum()} top3 labels")

if __name__ == '__main__':
    main()
