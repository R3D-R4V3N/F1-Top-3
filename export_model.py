# export_model.py

import argparse
import joblib

from train_model import build_and_train_pipeline as build_rf
from train_model_lgbm import build_and_train_pipeline as build_lgbm
from train_model_xgb import build_and_train_pipeline as build_xgb

def save_pipeline(algorithm: str = "rf"):
    """Train en bewaar het pipeline-model voor het gekozen algoritme."""

    if algorithm == "rf":
        pipeline = build_rf()
    elif algorithm == "lgbm":
        pipeline = build_lgbm()
    elif algorithm == "xgb":
        pipeline = build_xgb()
    else:
        raise ValueError(f"Onbekend algoritme: {algorithm}")

    # Sla op
    joblib.dump(pipeline, 'f1_top3_pipeline.joblib')
    print("Pipeline opgeslagen als f1_top3_pipeline.joblib")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exporteer een getrainde pipeline")
    parser.add_argument(
        "--algo",
        choices=["rf", "lgbm", "xgb"],
        default="rf",
        help="Welk algoritme moet worden getraind en opgeslagen"
    )
    args = parser.parse_args()
    save_pipeline(args.algo)
