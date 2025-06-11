# train_model_logreg.py

import pandas as pd
import numpy as np
from utils.time_series import GroupTimeSeriesSplit

from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    mean_absolute_error,
)


def build_and_train_pipeline(export_csv: bool = True, csv_path: str = "logreg_model_performance.csv"):
    """Train een LogisticRegression-model en retourneer het beste model en de hyperparameters.

    Parameters
    ----------
    export_csv : bool, optional
        Of de evaluatiemetrics naar ``csv_path`` weggeschreven moeten worden.
    csv_path : str, optional
        Pad waar de resultaten worden opgeslagen. Standaard ``logreg_model_performance.csv``
        zodat elk algoritme zijn eigen csv heeft.
    """

    # 1. Laad en sorteer data
    df = pd.read_csv('processed_data.csv', parse_dates=['date'])
    df = df.sort_values('date')

    # 2. Features
    numeric_feats = [
        'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
        'month', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
        'grid_diff', 'Q3_diff',
        'finish_rate_prev5',
        'team_qual_gap',
        'driver_points_prev', 'driver_rank_prev',
        'constructor_points_prev', 'constructor_rank_prev',

        # Overtakes-features
        'weighted_overtakes',
        'overtakes_per_lap',
        'weighted_overtakes_per_lap',
        'ewma_overtakes_per_lap',
        'ewma_weighted_overtakes_per_lap'
    ]
    categorical_feats = ['circuit_country', 'circuit_city']
    df['race_id'] = df['season'] * 100 + df['round']
    X = df[numeric_feats + categorical_feats]
    y = df['top3']
    groups = df['race_id'].values

    # 3. Train/test split op basis van race_id
    unique_races = df['race_id'].drop_duplicates()
    split_idx = int(len(unique_races) * 0.8)
    train_races = unique_races.iloc[:split_idx]
    test_races = unique_races.iloc[split_idx:]
    train_mask = df['race_id'].isin(train_races)
    test_mask = df['race_id'].isin(test_races)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    train_groups = groups[train_mask]

    # 4. Preprocessing
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
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

    # 5. Pipeline met LogisticRegression
    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))
    ])

    # 6. Hyperparameter grid
    param_grid = {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__class_weight': [None, 'balanced'],
    }

    # 7. GridSearchCV met GroupTimeSeriesSplit
    cv = GroupTimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X_train, y_train, groups=train_groups)

    # 7b. Learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        grid.best_estimator_, X, y,
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

    # 8. Resultaten
    print("=== Logistic Regression Best Params & CV ROC AUC ===")
    print(grid.best_params_)
    print(f"Best CV ROC AUC: {grid.best_score_:.3f}\n")

    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]
    print("=== Logistic Regression Test Performance ===")
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
            'Metric': ['CV ROC AUC', 'Test ROC AUC', 'Mean Abs Error', 'PR AUC'],
            'Value': [grid.best_score_, test_auc, mae, pr_auc],
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

    return grid.best_estimator_, grid.best_params_


def main():
    build_and_train_pipeline()


if __name__ == '__main__':
    main()
