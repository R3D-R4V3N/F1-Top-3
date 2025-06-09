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
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
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
    categorical_feats = ['circuit_country','circuit_city']
    X = df[numeric_feats + categorical_feats]
    y = df['top3']
    groups = df['race_id'].values

    # 2. Preprocessing pipelines
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
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

    # 3. Definieer algoritmen en hyperparametergrids
    algorithms = {
        'RandomForest': {
            'estimator': RandomForestClassifier(random_state=42),
            'param_grid': {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [None, 5],
                'clf__min_samples_split': [2, 5],
            },
        },
        'LightGBM': {
            'estimator': LGBMClassifier(random_state=42),
            'param_grid': {
                'clf__n_estimators': [200, 500],
                'clf__learning_rate': [0.05, 0.1],
                'clf__max_depth': [-1, 5],
            },
        },
        'CatBoost': {
            'estimator': CatBoostClassifier(random_state=42, verbose=0),
            'param_grid': {
                'clf__iterations': [200, 500],
                'clf__depth': [6, 8],
                'clf__learning_rate': [0.03, 0.1],
            },
        },
    }

    results = {}
    outer_cv = GroupTimeSeriesSplit(n_splits=6)

    for name, cfg in algorithms.items():
        pipe = Pipeline([
            ('pre', pre),
            ('clf', cfg['estimator'])
        ])
        inner_cv = GroupTimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(
            pipe,
            cfg['param_grid'],
            scoring='roc_auc',
            cv=inner_cv,
            n_jobs=-1,
        )

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
        results[name] = (scores.mean(), scores.std())
        print(f"{name} ROC AUC scores:", scores)
        print(f"{name} mean & std: {scores.mean():.4f} {scores.std():.4f}")

    if export_csv:
        data = {
            'Algorithm': [],
            'Mean': [],
            'Std': [],
        }
        for alg, (m, s) in results.items():
            data['Algorithm'].append(alg)
            data['Mean'].append(m)
            data['Std'].append(s)
        perf_df = pd.DataFrame(data).set_index('Algorithm')
        perf_df.to_csv(csv_path)
        print(f"Model performance saved to {csv_path}")

if __name__ == '__main__':
    main()
