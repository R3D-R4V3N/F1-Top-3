# train_model.py

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

from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    mean_absolute_error,
)

def build_and_train_pipeline(export_csv=True, csv_path="model_performance.csv"):
    """Bouwt de pipeline, traint hem en retourneert het beste model en de
    corresponderende hyperparameters.

    Parameters
    ----------
    export_csv : bool, optional
        Of de evaluatiemetrics naar ``csv_path`` weggeschreven moeten worden.
    csv_path : str, optional
        Pad waar het CSV-bestand met prestaties wordt opgeslagen.
    """

    # 1. Laad de verwerkte data en sorteer chronologisch
    df = pd.read_csv('processed_data.csv', parse_dates=['date'])
    df = df.sort_values('date')
    df['race_id'] = df['season'] * 100 + df['round']

    # 2. Definieer features & target
    numeric_feats = [
        'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
        'month', 'weekday','driver_age', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
        'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
        'air_temperature', 'track_temperature',
        'humidity', 'pressure', 'rainfall', 'wind_speed', 'wind_direction',
        'grid_diff', 'Q3_diff', 'grid_temp_int',
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
    categorical_feats = ['circuit_country','circuit_city']

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
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    train_groups = groups[train_mask]

    # 4. Preprocessing pipelines
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, categorical_feats)
    ])

    # 5. Full pipeline
    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # 6. Hyperparameter grid
    # Iets uitgebreidere grid om mogelijke overfitting beter te controleren
    param_grid = {
        'clf__n_estimators': [200, 500],
        'clf__max_depth':    [None, 5, 10, 20],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2'],
        'clf__class_weight': [None, 'balanced']
    }

    # 7. GridSearchCV met groepsgebaseerde tijdsplits
    cv = GroupTimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        pipe, param_grid,
        scoring='roc_auc',
        cv=cv, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train, groups=train_groups)

    # 7b. Learning curve to detect over- or underfitting
    train_sizes, train_scores, val_scores = learning_curve(
        grid.best_estimator_, X, y,
        groups=groups,
        cv=GroupTimeSeriesSplit(n_splits=5), scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    print("\nLearning curve (ROC AUC):")
    for sz, tr, val in zip(train_sizes, train_mean, val_mean):
        print(f"  {int(sz)} samples -> train {tr:.3f}, val {val:.3f}")

    # 8. Print performances
    print("Best parameters:", grid.best_params_)
    print(f"Best CV ROC AUC: {grid.best_score_:.3f}")

    y_pred  = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]
    print("\nTestset performance:")
    print(classification_report(y_test, y_pred))
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # **MAE erbij**
    mae = mean_absolute_error(y_test, y_proba)
    print(f"Mean Absolute Error (proba vs true): {mae:.3f}")

    # Confusion matrix, PR-AUC, misclassificaties …
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    print(f"Precision-Recall AUC: {pr_auc:.3f}")

    df_test = df.loc[X_test.index].copy()
    df_test['pred']  = y_pred
    df_test['proba'] = y_proba
    miscl = df_test[df_test['pred'] != df_test['top3']]
    print("\nVoorbeeld misclassificaties:")
    print(miscl[['Driver.driverId','raceName','finish_position','pred','proba']].head(5))

    if export_csv:
        base_metrics = {
            'Metric': ['CV ROC AUC', 'Test ROC AUC', 'Mean Abs Error', 'PR AUC'],
            'Value': [grid.best_score_, roc_auc_score(y_test, y_proba), mae, pr_auc]
        }

        # Breid uit met learning-curve resultaten
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

    # Return de uiteindelijke pipeline en de beste hyperparameters
    return grid.best_estimator_, grid.best_params_

def main():
    build_and_train_pipeline()

if __name__ == '__main__':
    main()
