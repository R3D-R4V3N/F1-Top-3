import pandas as pd
import numpy as np
from utils.time_series import GroupTimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
)

def main(export_csv=True, csv_path="model_performance.csv"):
    """Voert nested cross-validation uit en exporteert optioneel de resultaten."""

    # 1. Data laden en sorteren op datum
    df = pd.read_csv('processed_data.csv', parse_dates=['date'])
    df = df.sort_values('date')
    df['race_id'] = df['season'] * 100 + df['round']
    numeric_feats = [
        'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
        'month', 'weekday', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
        'air_temperature', 'track_temperature', 'humidity', 'pressure', 'rainfall',
        'wind_speed', 'wind_direction',
        'finish_rate_prev5',
        'driver_points_prev', 'driver_rank_prev',
        'constructor_points_prev', 'constructor_rank_prev',

        # Overtakes-features
        'overtakes_count',             # absolute aantal inhaalacties vorige races
        'weighted_overtakes',          # gewogen aantal inhaalacties
        'overtakes_per_lap',           # genormaliseerd per lap
        'weighted_overtakes_per_lap',   # gewogen én genormaliseerd
        'ewma_overtakes_per_lap',
        'ewma_weighted_overtakes_per_lap',
        'driver_age'
    ]
    categorical_feats = ['circuit_country','circuit_city']
    X = df[numeric_feats + categorical_feats]
    y = df['top3']
    groups = df['race_id'].values

    # 2. Preprocessing pipelines
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler',  StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore'))
    ])
    pre = ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, categorical_feats)
    ])

    # 3. Full pipeline + param grid
    pipe = Pipeline([
        ('pre', pre),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 5],
        'clf__min_samples_split': [2, 5],
        'clf__class_weight': [None, 'balanced']
    }

    # 4. Inner CV voor tuning met tijdreekssplits
    inner_cv = GroupTimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring='roc_auc',
        cv=inner_cv,
        n_jobs=-1,
    )

    # 5. Outer CV voor evaluatie met tijdreekssplits
    outer_cv = GroupTimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        train_groups = groups[train_idx]
        grid.fit(X_train, y_train, groups=train_groups)
        y_pred = grid.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_pred)
        scores.append(score)
    scores = np.array(scores)
    print("Nested CV ROC AUC scores:", scores)
    print("Mean & std:", scores.mean(), scores.std())

    if export_csv:
        perf_df = pd.DataFrame({
            'Metric': ['Nested CV ROC AUC Mean', 'Nested CV ROC AUC Std'],
            'Value': [scores.mean(), scores.std()]
        }).set_index('Metric')
        perf_df.to_csv(csv_path)
        print(f"Model performance saved to {csv_path}")

if __name__ == '__main__':
    main()
