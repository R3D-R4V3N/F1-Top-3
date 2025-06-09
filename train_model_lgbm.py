# train_model_lgbm.py

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    make_scorer,
    precision_recall_curve,
    auc,
    mean_absolute_error,
)

def build_and_train_pipeline(export_csv=True, csv_path="model_performance.csv", learning_curve=False):
    """Train een LightGBM-model en retourneer het beste model.

    Parameters
    ----------
    export_csv : bool, optional
        Of de evaluatiemetrics naar ``csv_path`` weggeschreven moeten worden.
    csv_path : str, optional
        Pad waar het CSV-bestand met prestaties wordt opgeslagen.
    learning_curve : bool, optional
        Bereken ook een learning curve met ROC-AUC scores.
    """

    # 1. Laad data
    df = pd.read_csv('processed_data.csv')

    # 2. Features & target
    numeric_feats = [
        'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
        'month', 'weekday', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
        'air_temperature', 'track_temperature',
        'overtakes_count', 'weighted_overtakes'
    ]
    categorical_feats = ['circuit_country','circuit_city']
    X = df[numeric_feats + categorical_feats]
    y = df['top3']

    # 3. Train/test-split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

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

    # 5. Pipeline met LightGBM
    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', LGBMClassifier(random_state=42))
    ])

    # 6. LightGBM hyperparameter grid
    param_grid = {
        'clf__n_estimators': [100, 300],
        'clf__learning_rate': [0.01, 0.1],
        'clf__num_leaves': [31, 63],
        'clf__max_depth': [-1, 10]
    }

    # 7. GridSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_train, y_train)

    # 8. Output beste resultaat
    print("=== LightGBM Best Params & CV ROC AUC ===")
    print(grid.best_params_)
    print(f"CV ROC AUC: {grid.best_score_:.3f}\n")

    # 9. Testset evaluatie
    y_pred  = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]
    print("=== LightGBM Test Performance ===")
    print(classification_report(y_test, y_pred))
    test_auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC AUC: {test_auc:.3f}")

    mae = mean_absolute_error(y_test, y_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)

    if export_csv:
        perf_df = pd.DataFrame({
            'Metric': ['CV ROC AUC', 'Test ROC AUC', 'Mean Abs Error', 'PR AUC'],
            'Value': [grid.best_score_, test_auc, mae, pr_auc]
        }).set_index('Metric')
        perf_df.to_csv(csv_path)

        if learning_curve:
            train_sizes = [0.25, 0.5, 0.75, 1.0]
            lc_sizes, lc_train, lc_val = learning_curve(
                grid.best_estimator_, X_train, y_train,
                cv=cv, scoring='roc_auc', train_sizes=train_sizes, n_jobs=-1
            )
            with open(csv_path, "a") as f:
                for size, t, v in zip(lc_sizes, lc_train, lc_val):
                    t_mean = t.mean()
                    v_mean = v.mean()
                    print(f"Train size {size}: train AUC {t_mean:.3f}, val AUC {v_mean:.3f}")
                    f.write(f"LC {size} Train ROC AUC,{t_mean}\n")
                    f.write(f"LC {size} Val ROC AUC,{v_mean}\n")

        print(f"Model performance saved to {csv_path}")

    return grid.best_estimator_

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument("--learning-curve", action="store_true",
                        help="Calculate learning curve metrics")
    args = parser.parse_args()
    build_and_train_pipeline(learning_curve=args.learning_curve)


if __name__ == '__main__':
    main()
