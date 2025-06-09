import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

FEATURE_COLS = [
    'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
    'month', 'weekday', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
    'air_temperature', 'track_temperature', 'grid_diff', 'Q3_diff', 'grid_temp_int',
    'driver_points_prev', 'driver_rank_prev',
    'constructor_points_prev', 'constructor_rank_prev',
    'circuit_country', 'circuit_city',
]


def main():
    """Load trained pipeline and processed dataset, then output feature importance."""
    df = pd.read_csv('processed_data.csv', parse_dates=['date'])
    pipeline = joblib.load('f1_top3_pipeline.joblib')

    # Use only feature columns present in the pipeline
    X = df[FEATURE_COLS]
    y = df['top3']

    result = permutation_importance(
        pipeline, X, y, n_repeats=5, random_state=42, n_jobs=-1
    )
    imp_df = (
        pd.DataFrame({'feature': FEATURE_COLS, 'importance': result.importances_mean})
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv('feature_importance.csv', index=False)
    print(imp_df)


if __name__ == '__main__':
    main()
