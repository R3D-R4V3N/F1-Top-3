# train_model_catboost.py

import pandas as pd
import numpy as np
try:
    from sklearn.model_selection import GroupTimeSeriesSplit
except ImportError:  # scikit-learn < 1.3
    class GroupTimeSeriesSplit:
        """Simple backport that keeps complete groups in each split."""

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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    mean_absolute_error,
)


def build_and_train_pipeline(export_csv: bool = True, csv_path: str = "model_performance.csv"):
    """Train een CatBoost-model en retourneer het beste model en de hyperparameters."""

    # 1. Data laden
    df = pd.read_csv('processed_data.csv', parse_dates=['date'])
    df = df.sort_values('date')
    df['race_id'] = df['season'] * 100 + df['round']

    # 2. Features en target
    numeric_feats = [
        'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
        'month', 'weekday', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
        'air_temperature', 'track_temperature',
        'driver_points_prev', 'driver_rank_prev',
        'constructor_points_prev', 'constructor_rank_prev',

        # Overtakes-features
        'overtakes_count',             # absolute aantal inhaalacties vorige races
        'weighted_overtakes',          # gewogen aantal inhaalacties
        'overtakes_per_lap',           # genormaliseerd per lap
        'weighted_overtakes_per_lap',   # gewogen én genormaliseerd
        'ewma_overtakes_per_lap',
        'ewma_weighted_overtakes_per_lap'
    ]
    categorical_feats = ['circuit_country', 'circuit_city']

    X = df[numeric_feats + categorical_feats]
    y = df['top3']

    # 3. Tijdgebaseerde split op basis van unieke races
    unique_races = df['race_id'].drop_duplicates()
    train_end = int(len(unique_races) * 0.7)
    val_end = int(len(unique_races) * 0.8)
    train_races = unique_races.iloc[:train_end]
    val_races = unique_races.iloc[train_end:val_end]
    test_races = unique_races.iloc[val_end:]

    train_mask = df['race_id'].isin(train_races)
    val_mask = df['race_id'].isin(val_races)
    test_mask = df['race_id'].isin(test_races)

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    groups = df['race_id'].values

    # 4. Preprocessing
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, categorical_feats)
    ])

    # Fit preprocessor and transform datasets
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    cat_indices = list(range(len(numeric_feats), len(numeric_feats) + len(categorical_feats)))

    # 5. Uitgebreidere hyperparameter grid
    param_grid = {
        'iterations': [500, 1000],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [3, 5, 7],
        'border_count': [64, 128],
    }

    best_score = -np.inf
    best_params = None
    best_model = None
    for params in ParameterGrid(param_grid):
        model = CatBoostClassifier(random_state=42, verbose=0, **params)
        model.fit(
            X_train_p,
            y_train,
            eval_set=(X_val_p, y_val),
            cat_features=cat_indices,
            early_stopping_rounds=50,
        )
        proba_val = model.predict_proba(X_val_p)[:, 1]
        auc_val = roc_auc_score(y_val, proba_val)
        if auc_val > best_score:
            best_score = auc_val
            best_params = params
            best_model = model

    # 6b. Learning curve met beste model (zonder early stopping)
    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', CatBoostClassifier(random_state=42, verbose=0, **best_params))
    ])
    train_sizes, train_scores, val_scores = learning_curve(
        pipe,
        X,
        y,
        groups=groups,
        cv=GroupTimeSeriesSplit(n_splits=5),
        scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    print("\nLearning curve (ROC AUC):")
    for sz, tr, val in zip(train_sizes, train_mean, val_mean):
        print(f"  {int(sz)} samples -> train {tr:.3f}, val {val:.3f}")

    print("=== CatBoost Best Params & Val ROC AUC ===")
    print(best_params)
    print(f"Validation ROC AUC: {best_score:.3f}\n")

    y_pred = best_model.predict(X_test_p)
    y_proba = best_model.predict_proba(X_test_p)[:, 1]
    print("=== CatBoost Test Performance ===")
    print(classification_report(y_test, y_pred))
    test_auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC AUC: {test_auc:.3f}")

    mae = mean_absolute_error(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)

    if export_csv:
        base_metrics = {
            'Metric': ['Val ROC AUC', 'Test ROC AUC', 'Mean Abs Error', 'PR AUC'],
            'Value': [best_score, test_auc, mae, pr_auc],
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

    # Fit final pipeline on train+val for export
    final_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', CatBoostClassifier(random_state=42, verbose=0, **best_params))
    ])
    final_pipe.fit(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val]),
        clf__cat_features=cat_indices,
    )

    return final_pipe, best_params


def main():
    build_and_train_pipeline()


if __name__ == '__main__':
    main()
