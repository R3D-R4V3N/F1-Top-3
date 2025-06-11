# train_model_stacking.py

"""Train a stacking ensemble using tuned base models.

This script loads the processed F1 dataset and reuses the same
preprocessing steps and ``GroupTimeSeriesSplit`` logic as the other
training scripts. Tuned RandomForest, LightGBM and XGBoost classifiers
are used as base learners and a logistic regression model acts as the
final estimator in a ``StackingClassifier``. The output mirrors the
metrics printed by the other ``train_model_*`` scripts so results can be
compared easily.
"""

import pandas as pd
import numpy as np
from utils.time_series import GroupTimeSeriesSplit
from utils import get_feature_lists, build_preprocessor

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    mean_absolute_error,
)
from sklearn.model_selection import (
    cross_val_score,
    learning_curve,
)

from train_model import build_and_train_pipeline as build_rf
from train_model_lgbm import build_and_train_pipeline as build_lgbm
from train_model_xgb import build_and_train_pipeline as build_xgb


def _extract_clf(pipeline):
    """Return the classifier from a fitted pipeline."""

    return pipeline.named_steps.get("clf")


def build_and_train_pipeline(export_csv: bool = True,
                             csv_path: str = "stacking_model_performance.csv"):
    """Train a stacking ensemble and optionally export metrics.

    Parameters
    ----------
    export_csv : bool, optional
        Whether to write the evaluation metrics to ``csv_path``.
    csv_path : str, optional
        Path to save the CSV file with the model's performance. By default this
        is ``stacking_model_performance.csv`` so that each model writes to its
        own dedicated file.
    """

    # 1. Load and sort the processed dataset
    df = pd.read_csv("processed_data.csv", parse_dates=["date"])
    df = df.sort_values("date")
    df["race_id"] = df["season"] * 100 + df["round"]

    # 2. Feature lists and target
    numeric_feats, categorical_feats = get_feature_lists()

    X = df[numeric_feats + categorical_feats]
    y = df["top3"]
    groups = df["race_id"].values

    # 3. Split train and test based on entire races
    unique_races = df["race_id"].drop_duplicates()
    split_idx = int(len(unique_races) * 0.8)
    train_races = unique_races.iloc[:split_idx]
    test_races = unique_races.iloc[split_idx:]
    train_mask = df["race_id"].isin(train_races)
    test_mask = df["race_id"].isin(test_races)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    train_groups = groups[train_mask]

    # 4. Shared preprocessing
    preprocessor = build_preprocessor()

    # 5. Obtain tuned base estimators
    rf_pipe, _ = build_rf(export_csv=False)
    lgbm_pipe, _ = build_lgbm(export_csv=False)
    xgb_pipe, _ = build_xgb(export_csv=False)

    base_learners = [
        ("rf", _extract_clf(rf_pipe)),
        ("lgbm", _extract_clf(lgbm_pipe)),
        ("xgb", _extract_clf(xgb_pipe)),
    ]

    final_est = LogisticRegression(max_iter=1000)

    stacker = StackingClassifier(
        estimators=base_learners,
        final_estimator=final_est,
        passthrough=False,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", stacker),
    ])

    # 6. Cross-validation on training data
    cv = GroupTimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        groups=train_groups,
        scoring="roc_auc",
        n_jobs=-1,
    )
    print("=== Stacking CV ROC AUC ===")
    print(cv_scores)
    print(f"Mean CV ROC AUC: {cv_scores.mean():.3f}\n")

    # 7. Fit on training set
    pipeline.fit(X_train, y_train)

    # 7b. Learning curve using the entire dataset
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        groups=groups,
        cv=GroupTimeSeriesSplit(n_splits=5),
        scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    print("Learning curve (ROC AUC):")
    for sz, tr, val in zip(train_sizes, train_mean, val_mean):
        print(f"  {int(sz)} samples -> train {tr:.3f}, val {val:.3f}")

    # 8. Evaluate on the hold-out test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print("\n=== Stacking Test Performance ===")
    print(classification_report(y_test, y_pred))
    test_auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC AUC: {test_auc:.3f}")

    mae = mean_absolute_error(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    print(f"Precision-Recall AUC: {pr_auc:.3f}")

    if export_csv:
        base_metrics = {
            "Metric": ["CV ROC AUC", "Test ROC AUC", "Mean Abs Error", "PR AUC"],
            "Value": [cv_scores.mean(), test_auc, mae, pr_auc],
        }

        lc_metrics = []
        lc_values = []
        for sz, tr, val in zip(train_sizes, train_mean, val_mean):
            lc_metrics.append(f"LC {int(sz)} Train ROC AUC")
            lc_values.append(tr)
            lc_metrics.append(f"LC {int(sz)} Val ROC AUC")
            lc_values.append(val)

        all_metrics = base_metrics["Metric"] + lc_metrics
        all_values = base_metrics["Value"] + lc_values

        perf_df = (
            pd.DataFrame({"Metric": all_metrics, "Value": all_values})
            .set_index("Metric")
        )
        perf_df.to_csv(csv_path)
        print(f"Model performance and learning curve saved to {csv_path}")

    return pipeline


def main():
    build_and_train_pipeline()


if __name__ == "__main__":
    main()
