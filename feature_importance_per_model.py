import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score

from export_model import save_pipeline

ALGORITHMS = ["catb"]


def manual_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    """Compute permutation importance using ROC AUC without relying on
    `permutation_importance`'s scoring logic.

    This avoids issues when the loaded pipeline does not expose a default
    `score` method that is compatible with the scorer utility.
    """

    rng = np.random.RandomState(random_state)
    baseline = roc_auc_score(y, model.predict_proba(X)[:, 1])
    importances = np.zeros((X.shape[1], n_repeats))

    for n in range(n_repeats):
        for i, col in enumerate(X.columns):
            X_permuted = X.copy()
            X_permuted[col] = rng.permutation(X_permuted[col].values)
            permuted_score = roc_auc_score(
                y, model.predict_proba(X_permuted)[:, 1]
            )
            importances[i, n] = baseline - permuted_score

    return importances.mean(axis=1), importances.std(axis=1)

print("Loading processed_data.csv...")
df = pd.read_csv('processed_data.csv', parse_dates=['date'])
df = df.sort_values('date')
df['race_id'] = df['season'] * 100 + df['round']

numeric_feats = [
    'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
    'month', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
    'grid_diff', 'Q3_diff',
    'weighted_overtakes',
    'overtakes_per_lap',
    'finish_rate_prev5',
    'team_qual_gap',
    'weighted_overtakes_per_lap',
    
    'ewma_overtakes_per_lap',
    'ewma_weighted_overtakes_per_lap'
]
categorical_feats = ['circuit_country', 'circuit_city']

X = df[numeric_feats + categorical_feats]
y = df['top3']

unique_races = df['race_id'].drop_duplicates()
split_idx = int(len(unique_races) * 0.8)
train_races = unique_races.iloc[:split_idx]
test_races = unique_races.iloc[split_idx:]
test_mask = df['race_id'].isin(test_races)
X_test = X[test_mask]
y_test = y[test_mask]

os.makedirs('feature_importances', exist_ok=True)

for algo in ALGORITHMS:
    print(f"\nTraining and exporting {algo} model...")
    save_pipeline()
    pipeline = joblib.load('f1_top3_pipeline.joblib')

    print(f"Computing permutation importance for {algo}...")
    means, stds = manual_permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=42
    )

    importance_df = (
        pd.DataFrame({
            'feature': X_test.columns,
            'importance_mean': means,
            'importance_std': stds,
        })
        .sort_values('importance_mean', ascending=False)
    )

    csv_path = f'feature_importances/{algo}_feature_importance.csv'
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
