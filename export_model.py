# export_model.py

import argparse
import joblib
from sklearn.base import clone
import pandas as pd

from utils.threshold import find_best_threshold

from train_model import build_and_train_pipeline as build_rf
from train_model_lgbm import build_and_train_pipeline as build_lgbm
from train_model_xgb import build_and_train_pipeline as build_xgb
from train_model_catboost import build_and_train_pipeline as build_catb
from train_model_logreg import build_and_train_pipeline as build_logreg
from train_model_stacking import build_and_train_pipeline as build_stack

def save_pipeline(algorithm: str = "rf"):
    """Train en bewaar het pipeline-model voor het gekozen algoritme."""

    if algorithm == "rf":
        pipeline, best_params = build_rf()
    elif algorithm == "lgbm":
        pipeline, best_params = build_lgbm()
    elif algorithm == "xgb":
        pipeline, best_params = build_xgb()
    elif algorithm == "catb":
        pipeline, best_params = build_catb()
    elif algorithm == "logreg":
        pipeline, best_params = build_logreg()
    elif algorithm == "stack":
        pipeline, best_params = build_stack()
    else:
        raise ValueError(f"Onbekend algoritme: {algorithm}")

    print("Beste hyperparameters:", best_params)

    # Fit opnieuw op volledige dataset en bepaal optimale drempel
    df = pd.read_csv("processed_data.csv", parse_dates=["date"])
    df = df.sort_values("date")
    df["race_id"] = df["season"] * 100 + df["round"]
    X_full = df.drop(columns=["top3"])
    y_full = df["top3"]
    groups = df["race_id"].values

    threshold = find_best_threshold(pipeline, X_full, y_full, groups, metric="pr")

    final_model = clone(pipeline)
    final_model.fit(X_full, y_full)

    joblib.dump({"model": final_model, "threshold": threshold}, "f1_top3_pipeline.joblib")
    print("Pipeline opgeslagen als f1_top3_pipeline.joblib")
    print(f"Optimale drempel: {threshold:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exporteer een getrainde pipeline")
    parser.add_argument(
        "--algo",
        choices=["rf", "lgbm", "xgb", "catb", "logreg", "stack"],
        default="rf",
        help="Welk algoritme moet worden getraind en opgeslagen"
    )
    args = parser.parse_args()
    save_pipeline(args.algo)
