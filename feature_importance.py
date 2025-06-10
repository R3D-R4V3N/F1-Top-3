import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

# Load processed data
print("Loading processed_data.csv...")
df = pd.read_csv('processed_data.csv', parse_dates=['date'])
df = df.sort_values('date')
df['race_id'] = df['season'] * 100 + df['round']

# Feature columns identical to training scripts
numeric_feats = [
    'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
    'month', 'weekday', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
    'air_temperature', 'track_temperature', 'humidity', 'pressure', 'rainfall',
    'wind_speed', 'wind_direction',
    'grid_diff', 'Q3_diff', 'grid_temp_int',
    'driver_points_prev', 'driver_rank_prev',
    'constructor_points_prev', 'constructor_rank_prev',
    'overtakes_count',
    'weighted_overtakes',
    'overtakes_per_lap',
    'weighted_overtakes_per_lap',
    'ewma_overtakes_per_lap',
    'ewma_weighted_overtakes_per_lap'
]
categorical_feats = ['circuit_country', 'circuit_city']

X = df[numeric_feats + categorical_feats]
y = df['top3']

# Split by race (20% of races as test set)
unique_races = df['race_id'].drop_duplicates()
split_idx = int(len(unique_races) * 0.8)
train_races = unique_races.iloc[:split_idx]
test_races = unique_races.iloc[split_idx:]
test_mask = df['race_id'].isin(test_races)
X_test = X[test_mask]
y_test = y[test_mask]

print("Loading trained pipeline...")
pipeline = joblib.load('f1_top3_pipeline.joblib')

print("Computing permutation importance on test split...")
result = permutation_importance(
    pipeline, X_test, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)

importance_df = (
    pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    })
    .sort_values('importance_mean', ascending=False)
)

print("\nPermutation importance (descending):")
print(importance_df.to_string(index=False))

importance_df.to_csv('feature_importance_global.csv', index=False)
print("\nSaved feature_importance_global.csv")
