# train_model_xgb.py

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    learning_curve,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    mean_absolute_error,
)


def build_and_train_pipeline(export_csv=True, csv_path="model_performance.csv"):
    # 1. Load processed data
    df = pd.read_csv('processed_data.csv')

    # 2. Define features and target
    numeric_feats = [
        'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
        'month', 'weekday', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
        'air_temperature', 'track_temperature', 'grid_diff', 'Q3_diff', 'grid_temp_int',
        'overtakes_count', 'weighted_overtakes'
    ]
    categorical_feats = ['circuit_country', 'circuit_city']

    X = df[numeric_feats + categorical_feats]
    y = df['top3']

    # 3. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Preprocessing
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

    # 5. Pipeline with XGBoost
    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    # 6. Hyperparameter grid for XGBoost
    param_grid = {
        'clf__n_estimators':     [100, 200, 300, 500],
        'clf__max_depth':        [3, 4, 5],
        'clf__subsample':        [0.6, 0.8, 1.0],
        'clf__colsample_bytree': [0.6, 0.8, 1.0],
        'clf__reg_lambda':       [0, 1, 5],
        'clf__reg_alpha':        [0, 0.5, 1],
        'clf__min_child_weight': [5, 10, 20]
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

    # 8. Learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        grid.best_estimator_, X, y,
        cv=cv,
        scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    print("\nLearning curve (ROC AUC):")
    for sz, tr, val in zip(train_sizes, train_mean, val_mean):
        print(f"  {int(sz)} samples -> train {tr:.3f}, val {val:.3f}")

    # 9. Best parameters & CV score
    print("=== XGBoost Best Params & CV ROC AUC ===")
    print(grid.best_params_)
    print(f"Best CV ROC AUC: {grid.best_score_:.3f}\n")

    # 10. Extract best parameters and retrain with early stopping
    best_params = grid.best_params_
    clf_params = {k.replace('clf__',''): v for k, v in best_params.items()}
    clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        **clf_params
    )
    # Preprocess data separately
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans  = preprocessor.transform(X_test)
    try:
        clf.fit(
            X_train_trans,
            y_train,
            eval_set=[(X_test_trans, y_test)],
            callbacks=[xgb.callback.EarlyStopping(rounds=50)],
            verbose=False,
        )
    except TypeError:
        try:
            # Older XGBoost versions without callback API
            clf.fit(
                X_train_trans,
                y_train,
                eval_set=[(X_test_trans, y_test)],
                early_stopping_rounds=50,
                verbose=False,
            )
        except TypeError:
            # Fallback for very old versions without early stopping support
            clf.fit(
                X_train_trans,
                y_train,
                eval_set=[(X_test_trans, y_test)],
                verbose=False,
            )

    # 11. Test evaluation
    y_pred  = clf.predict(X_test_trans)
    y_proba = clf.predict_proba(X_test_trans)[:, 1]
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

    # 12. Save performance
    if export_csv:
        base_metrics = {
            'Metric': ['CV ROC AUC', 'Test ROC AUC', 'Mean Abs Error', 'PR AUC'],
            'Value': [grid.best_score_, test_auc, mae, pr_auc],
        }
        lc_metrics, lc_values = [], []
        for sz, tr, val in zip(train_sizes, train_mean, val_mean):
            lc_metrics += [f'LC {int(sz)} Train ROC AUC', f'LC {int(sz)} Val ROC AUC']
            lc_values += [tr, val]
        perf_df = pd.DataFrame({'Metric': base_metrics['Metric'] + lc_metrics,
                                'Value':  base_metrics['Value']  + lc_values})
        perf_df.set_index('Metric').to_csv(csv_path)
        print(f"Model performance saved to {csv_path}")

    return clf


def main():
    build_and_train_pipeline()

if __name__ == '__main__':
    main()
