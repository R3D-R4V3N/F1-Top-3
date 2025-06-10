# train_model_lgbm.py

import argparse
import pandas as pd
import numpy as np
try:
    from sklearn.model_selection import GroupTimeSeriesSplit
except ImportError:  # scikit-learn < 1.3
    class GroupTimeSeriesSplit:
        def __init__(self, n_splits: int = 5, groups=None):
            self.n_splits = n_splits
            self.groups = np.array(groups) if groups is not None else None

        def split(self, X, y=None, groups=None):
            if groups is None:
                if self.groups is None:
                    raise ValueError("The 'groups' parameter is required")
                groups = self.groups
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
from sklearn.model_selection import (
    GridSearchCV,
    learning_curve,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    make_scorer,
    confusion_matrix,
    precision_recall_curve,
    auc,
    mean_absolute_error,
)
from sklearn.calibration import CalibratedClassifierCV

def build_and_train_pipeline(
    export_csv: bool = True,
    csv_path: str = "model_performance.csv",
    calib_csv_path: str = "model_performance_calibrated.csv",
):
    """Train een LightGBM-model en retourneer het beste model, de
    hyperparameters en de best gevonden iteratie.

    Parameters
    ----------
    export_csv : bool, optional
        Of de evaluatiemetrics naar ``csv_path`` weggeschreven moeten worden.
    csv_path : str, optional
        Pad waar het CSV-bestand met prestaties wordt opgeslagen.
    """

    # 1. Laad data en sorteer chronologisch
    df = pd.read_csv('processed_data.csv', parse_dates=['date'])
    df = df.sort_values('date')
    df['race_id'] = df['season'] * 100 + df['round']

    # 2. Features & target
    numeric_feats = [
        'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
        'month', 'weekday', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
        'air_temperature', 'track_temperature',
        'driver_points_prev', 'driver_rank_prev',
        'constructor_points_prev', 'constructor_rank_prev',
        'driver_home_race', 'rank_diff',

        # Overtakes-features
        'overtakes_count',             # absolute aantal inhaalacties vorige races
        'weighted_overtakes',          # gewogen aantal inhaalacties
        'overtakes_per_lap',           # genormaliseerd per lap
        'weighted_overtakes_per_lap',   # gewogen én genormaliseerd
        'ewma_overtakes_per_lap',
        'ewma_weighted_overtakes_per_lap'
    ]
    categorical_feats = ['circuit_country','circuit_city','track_type']
    X = df[numeric_feats + categorical_feats]
    y = df['top3']
    groups = df['race_id'].values

    # 3. Tijdgebaseerde train/test-split (laatste 20% als test) op racebasis
    unique_races = df['race_id'].drop_duplicates()
    split_idx = int(len(unique_races) * 0.8)
    train_races = unique_races.iloc[:split_idx]
    test_races = unique_races.iloc[split_idx:]
    train_mask = df['race_id'].isin(train_races)
    test_mask = df['race_id'].isin(test_races)
    X_train_full, X_test = X[train_mask], X[test_mask]
    y_train_full, y_test = y[train_mask], y[test_mask]
    groups_full = groups[train_mask]

    # 3b. Validation split binnen training met GroupTimeSeriesSplit
    gts = GroupTimeSeriesSplit(n_splits=5)
    train_idx, val_idx = list(gts.split(X_train_full, y_train_full, groups=groups_full))[-1]
    X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
    train_groups = groups_full[train_idx]

    # 4. Preprocessing
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ])

    # fit preprocessing on training data and transform all splits
    preprocessor.fit(X_train_full)
    X_train_full_tr = preprocessor.transform(X_train_full)
    X_test_tr = preprocessor.transform(X_test)
    X_train_tr = preprocessor.transform(X_train)
    X_val_tr = preprocessor.transform(X_val)
    X_all_tr = preprocessor.transform(X)

    # 5. LightGBM model zonder pipeline
    clf = LGBMClassifier(random_state=42)

    # 6. Hyperparameter grid
    pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    param_grid = {
        'n_estimators': [200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'max_depth': [-1, 5, 10],
        'min_child_samples': [20, 40],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_lambda': [0.0, 0.1, 1.0],
        'scale_pos_weight': [1, pos_weight]
    }

    # 7. GridSearchCV met groepsgebaseerde tijdsplits
    cv = GroupTimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=2
    )
    grid.fit(
        X_train_tr,
        y_train,
        groups=train_groups,
        eval_set=[(X_val_tr, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
        ],
    )
    best_iter = grid.best_estimator_.best_iteration_

    # 7b. Learning curve to detect over- or underfitting
    train_sizes, train_scores, val_scores = learning_curve(
        grid.best_estimator_,
        X_all_tr,
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

    # 8. Output beste resultaat
    print("=== LightGBM Best Params & CV ROC AUC ===")
    print(grid.best_params_)
    print(f"CV ROC AUC: {grid.best_score_:.3f}")
    print(f"Best Iteration: {best_iter}\n")

    # 9. Testset evaluatie
    y_pred  = grid.predict(X_test_tr)
    y_proba = grid.predict_proba(X_test_tr)[:, 1]
    print("=== LightGBM Test Performance ===")
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
            'Metric': [
                'CV ROC AUC',
                'Test ROC AUC',
                'Mean Abs Error',
                'PR AUC',
                'Best Iteration',
            ],
            'Value': [grid.best_score_, test_auc, mae, pr_auc, best_iter],
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

    # Calibrate the best model on the training data
    calibrator = CalibratedClassifierCV(
        estimator=grid.best_estimator_,
        method='isotonic',
        cv=GroupTimeSeriesSplit(n_splits=5, groups=train_groups),
    )
    calibrator.fit(X_train_tr, y_train)

    # Evaluate calibrated model on the held-out test set
    y_proba_cal = calibrator.predict_proba(X_test_tr)[:, 1]
    test_auc_cal = roc_auc_score(y_test, y_proba_cal)
    mae_cal = mean_absolute_error(y_test, y_proba_cal)
    prec_cal, recall_cal, _ = precision_recall_curve(y_test, y_proba_cal)
    pr_auc_cal = auc(recall_cal, prec_cal)

    print("\n=== Calibrated Test Performance ===")
    print(f"ROC AUC: {test_auc_cal:.3f}")
    print(f"PR AUC: {pr_auc_cal:.3f}")
    print(f"MAE: {mae_cal:.3f}")

    if export_csv:
        calib_df = pd.DataFrame(
            {
                'Metric': ['ROC AUC', 'PR AUC', 'Mean Abs Error'],
                'Value': [test_auc_cal, pr_auc_cal, mae_cal],
            }
        ).set_index('Metric')
        calib_df.to_csv(calib_csv_path)
        print(f"Calibrated model performance saved to {calib_csv_path}")

    return calibrator, grid.best_params_

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument(
        "--csv",
        default="model_performance_lgbm.csv",
        help="Output CSV file for metrics",
    )
    args = parser.parse_args()
    build_and_train_pipeline(csv_path=args.csv)


if __name__ == '__main__':
    main()
