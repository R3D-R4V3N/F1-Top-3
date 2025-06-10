# infer.py

import pandas as pd
import joblib

# Inference script: use only data up to the chosen race date to predict top-3 finishers

def inference_for_date(cutoff_date):
    # 1. Load all processed data
    df_full = pd.read_csv('processed_data.csv', parse_dates=['date'])
    # 2. Keep only data up to and including cutoff_date
    df = df_full[df_full['date'] <= cutoff_date].copy()
    # 3. Sort by date to ensure rolling features use past data
    df = df.sort_values('date')
    df['Driver.dateOfBirth'] = pd.to_datetime(df['Driver.dateOfBirth'])

    # 4. Recompute rolling and interaction features on this subset
    df['avg_finish_pos'] = (
        df.groupby('Driver.driverId')['finish_position']
          .transform(lambda x: x.shift().expanding().mean())
    )
    df['avg_grid_pos'] = (
        df.groupby('Driver.driverId')['grid_position']
          .transform(lambda x: x.shift().expanding().mean())
    )
    df['avg_const_finish'] = (
        df.groupby('constructorId')['finish_position']
          .transform(lambda x: x.shift().expanding().mean())
    )
    df['grid_diff'] = df['avg_grid_pos'] - df['grid_position']
    df['driver_avg_Q3'] = (
        df.groupby('Driver.driverId')['Q3_sec']
          .transform(lambda x: x.shift().expanding().mean())
    )
    df['Q3_diff'] = df['driver_avg_Q3'] - df['Q3_sec']
    # Removed weather & standings based features

    # Impute missing values using only past observations

    for sec in ['Q1_sec', 'Q2_sec', 'Q3_sec']:
        run_med = (
            df.groupby('Circuit.circuitId')[sec]
              .transform(lambda s: s.expanding().median().shift())
        )
        global_med = df[sec].expanding().median().shift()
        df[sec] = df[sec].fillna(run_med)
        df[sec] = df[sec].fillna(global_med)
        df[sec] = df[sec].fillna(0)

    for col in ['avg_finish_pos', 'avg_grid_pos', 'avg_const_finish']:
        run_med = df[col].expanding().median().shift()
        df[col] = df[col].fillna(run_med)
        df[col] = df[col].fillna(0)


    df['qual_best_sec'] = df[['Q1_sec', 'Q2_sec', 'Q3_sec']].min(axis=1, skipna=True)
    team_best = df.groupby(['season', 'round', 'constructorId'])['qual_best_sec'].transform('min')
    df['team_qual_gap'] = (df['qual_best_sec'] - team_best).fillna(0)
    df.drop(columns=['qual_best_sec'], inplace=True)

    # 5. Select only the rows of the cutoff_date for testing
    df_test = df[df['date'] == cutoff_date].copy()

    # 6. Load the serialized pipeline
    pipeline = joblib.load('f1_top3_pipeline.joblib')

    # 7. Feature columns exactly as trained
    feature_cols = [
        'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
        'month', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
        'grid_diff', 'Q3_diff',
        'team_qual_gap',

        'num_pitstops',
        'avg_pitstop_duration',
        'tyre_degradation_rate',
        'qual_delta',
        'circuit_top3_freq',
        'head_to_head_vs_teammate',

        'circuit_country', 'circuit_city',
        # Overtakes-features
        'weighted_overtakes',
        'overtakes_per_lap',
        'weighted_overtakes_per_lap',
        'ewma_overtakes_per_lap',
        'ewma_weighted_overtakes_per_lap'
    ]
    X_test = df_test[feature_cols]

    # 8. Predict probabilities and apply threshold
    proba = pipeline.predict_proba(X_test)[:, 1]
    df_test['top3_proba'] = proba
    df_test['top3_pred']  = proba >= 0.41

    # 9. Show top-3 unique drivers
    top3 = (
        df_test[['Driver.driverId','raceName','top3_proba']]
        .sort_values('top3_proba', ascending=False)
        .drop_duplicates('Driver.driverId')
        .head(3)
    )

    print(f"Top-3 voorspellingen voor race op {cutoff_date.date()}:\n")
    print(top3.to_string(index=False))

if __name__ == '__main__':
    # Replace with desired race date for prediction/backtest
    inference_for_date(pd.Timestamp("2025-06-01"))
