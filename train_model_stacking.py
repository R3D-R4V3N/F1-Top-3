"""Train a stacking ensemble to predict top-3 finishes."""

import pandas as pd
import numpy as np

try:
    from sklearn.model_selection import GroupTimeSeriesSplit, learning_curve
except ImportError:  # scikit-learn < 1.3
    class GroupTimeSeriesSplit:
        def __init__(self, n_splits: int = 5):
            self.n_splits = n_splits
        def split(self, X, y=None, groups=None):
            if groups is None:
                raise ValueError("The 'groups' parameter is required")
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
            test_size = n_groups // (self.n_splits + 1)
            for i in range(self.n_splits):
                train_end = test_size * (i + 1)
                test_end = test_size * (i + 2)
                train_groups = unique_groups[:train_end]
                test_groups = unique_groups[train_end:test_end]
                train_idx = np.where(np.isin(groups, train_groups))[0]
                test_idx = np.where(np.isin(groups, test_groups))[0]
                yield train_idx, test_idx
        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits
    from sklearn.model_selection import learning_curve

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    mean_absolute_error,
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def build_and_train_pipeline(export_csv=True, csv_path="model_performance.csv"):
    df = pd.read_csv('processed_data.csv', parse_dates=['date'])
    df = df.sort_values('date')
    df['race_id'] = df['season'] * 100 + df['round']

    numeric_feats = [
        'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
        'month', 'weekday', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
        'air_temperature', 'track_temperature', 'grid_diff', 'Q3_diff', 'grid_temp_int',
        'driver_points_prev', 'driver_rank_prev',
        'constructor_points_prev', 'constructor_rank_prev',
        'overtakes_count', 'weighted_overtakes', 'overtakes_per_lap',
        'weighted_overtakes_per_lap', 'ewma_overtakes_per_lap',
        'ewma_weighted_overtakes_per_lap'
    ]
    categorical_feats = ['circuit_country', 'circuit_city']

    X = df[numeric_feats + categorical_feats]
    y = df['top3']
    groups = df['race_id'].values

    unique_races = df['race_id'].drop_duplicates()
    split_idx = int(len(unique_races) * 0.8)
    train_races = unique_races.iloc[:split_idx]
    test_races = unique_races.iloc[split_idx:]
    train_mask = df['race_id'].isin(train_races)
    test_mask = df['race_id'].isin(test_races)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    train_groups = groups[train_mask]

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, categorical_feats)
    ])

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    base_models = [
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
        ('lgbm', LGBMClassifier(random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ]
    cv = GroupTimeSeriesSplit(n_splits=5)

    train_meta = []
    test_meta = []
    for _, model in base_models:
        proba_cv = np.zeros_like(y_train, dtype=float)
        for train_idx, val_idx in cv.split(X_train_proc, y_train, train_groups):
            m = model.__class__(**model.get_params())
            m.fit(X_train_proc[train_idx], y_train.iloc[train_idx])
            proba_cv[val_idx] = m.predict_proba(X_train_proc[val_idx])[:, 1]
        model.fit(X_train_proc, y_train)
        proba_test = model.predict_proba(X_test_proc)[:, 1]
        train_meta.append(proba_cv)
        test_meta.append(proba_test)

    X_train_meta = np.column_stack(train_meta)
    X_test_meta = np.column_stack(test_meta)

    final_est = LogisticRegression(max_iter=1000)
    final_est.fit(X_train_meta, y_train)

    train_sizes, train_scores, val_scores = learning_curve(
        final_est, X_train_meta, y_train,
        groups=train_groups,
        cv=cv,
        scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    y_pred = final_est.predict(X_test_meta)
    y_proba = final_est.predict_proba(X_test_meta)[:, 1]

    print("=== Stacking Test Performance ===")
    print(classification_report(y_test, y_pred))
    test_auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC AUC: {test_auc:.3f}")

    mae = mean_absolute_error(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    print(f"PR AUC: {pr_auc:.3f}")

    if export_csv:
        base_metrics = {
            'Metric': ['CV ROC AUC', 'Test ROC AUC', 'Mean Abs Error', 'PR AUC'],
            'Value': [np.mean(val_mean), test_auc, mae, pr_auc],
        }
        lc_metrics = []
        lc_values = []
        for sz, tr, val in zip(train_sizes, train_mean, val_mean):
            lc_metrics.append(f'LC {int(sz)} Train ROC AUC')
            lc_values.append(tr)
            lc_metrics.append(f'LC {int(sz)} Val ROC AUC')
            lc_values.append(val)
        all_metrics = base_metrics['Metric'] + lc_metrics
        all_values = base_metrics['Value'] + lc_values
        perf_df = pd.DataFrame({'Metric': all_metrics, 'Value': all_values}).set_index('Metric')
        perf_df.to_csv(csv_path)
        print(f"Model performance and learning curve saved to {csv_path}")

    return final_est


def main():
    build_and_train_pipeline()

if __name__ == '__main__':
    main()
