# train_model_xgb.py

import pandas as pd
import numpy as np
try:
    from sklearn.model_selection import GroupTimeSeriesSplit
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
from sklearn.model_selection import (
    GridSearchCV,
    learning_curve,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    make_scorer,
    confusion_matrix,
    precision_recall_curve,
    auc,
    mean_absolute_error,
)


def build_and_train_pipeline(export_csv=True, csv_path="model_performance.csv"):
    """Train een XGBoost-model en retourneer het beste model samen met de
    optimale hyperparameters.

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

    # 2. Definieer features en target
    numeric_feats = [
        'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
        'month', 'weekday', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
        'air_temperature', 'track_temperature', 'grid_diff', 'Q3_diff', 'grid_temp_int',
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
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ])

    # 5. Pipeline met XGBoost
    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    # 6. Hyperparameter grid voor XGBoost
    # Ruimer grid met regularisatie om overfitting tegen te gaan
    param_grid = {
        'clf__n_estimators': [200, 400],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__subsample': [0.6, 0.8, 1.0],
        'clf__colsample_bytree': [0.6, 0.8, 1.0],
        'clf__gamma': [0, 0.1, 0.2, 0.3],
        'clf__min_child_weight': [1, 5, 10],
        'clf__reg_lambda': [1.0, 1.5],
        'clf__reg_alpha': [0.0, 0.1, 1.0]
    }

    # 7. GridSearchCV met groepsgebaseerde tijdsplits
    cv = GroupTimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_train, y_train, groups=train_groups)

    # 7b. Learning curve to detect over- or underfitting
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

    # 8. Beste parameters & CV-score
    print("=== XGBoost Best Params & CV ROC AUC ===")
    print(grid.best_params_)
    print(f"Best CV ROC AUC: {grid.best_score_:.3f}\n")

    # 8b. Train het beste model opnieuw met early stopping
    val_split = int(len(train_races) * 0.9)
    es_train_races = train_races.iloc[:val_split]
    es_val_races = train_races.iloc[val_split:]

    es_train_mask = df['race_id'].isin(es_train_races)
    es_val_mask = df['race_id'].isin(es_val_races)

    X_train_es, y_train_es = X[es_train_mask], y[es_train_mask]
    X_val_es, y_val_es = X[es_val_mask], y[es_val_mask]

    preprocessor.fit(X_train_es)
    X_train_es_t = preprocessor.transform(X_train_es)
    X_val_es_t = preprocessor.transform(X_val_es)
    X_test_t = preprocessor.transform(X_test)

    best_params = {k.split('__')[1]: v for k, v in grid.best_params_.items()}
    best_clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        **best_params
    )
    best_clf.fit(
        X_train_es_t,
        y_train_es,
        eval_set=[(X_val_es_t, y_val_es)],
        early_stopping_rounds=50,
        verbose=False,
    )

    # 9. Testset evaluatie
    y_pred  = best_clf.predict(X_test_t)
    y_proba = best_clf.predict_proba(X_test_t)[:, 1]
    print("=== XGBoost Test Performance ===")
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

    return best_clf, grid.best_params_

def main():
    build_and_train_pipeline()


if __name__ == '__main__':
    main()
