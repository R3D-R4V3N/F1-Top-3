# prepare_data.py

import pandas as pd


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


    # 15. Impute weather
    for col in ['air_temperature','track_temperature']:
        df[col] = df[col].fillna(df[col].median())

    # Drop helper cols
    df.drop(columns=['date_only','session_key'], inplace=True)

    # 16. Wegschrijven
    df.to_csv('processed_data.csv', index=False)
    print(f"processed_data.csv saved — {len(df)} rows, {df['top3'].sum()} top3 labels")

if __name__ == '__main__':
    main()
