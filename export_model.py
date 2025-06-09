# export_model.py

import argparse
import joblib
from sklearn.base import clone
import pandas as pd

from train_model import build_and_train_pipeline as build_rf
from train_model_lgbm import build_and_train_pipeline as build_lgbm
from train_model_xgb import build_and_train_pipeline as build_xgb

def save_pipeline(algorithm: str = "rf"):
    """Train en bewaar het pipeline-model voor het gekozen algoritme."""

    if algorithm == "rf":
        pipeline, best_params = build_rf()
    elif algorithm == "lgbm":
        pipeline, best_params = build_lgbm()
    elif algorithm == "xgb":
        pipeline, best_params = build_xgb()
    else:
        raise ValueError(f"Onbekend algoritme: {algorithm}")

    print("Beste hyperparameters:", best_params)

    # Fit opnieuw op volledige dataset met deze parameters
    df = pd.read_csv('processed_data.csv')
    X_full = df.drop(columns=['top3'])
    y_full = df['top3']

    final_model = clone(pipeline)
    final_model.fit(X_full, y_full)

    joblib.dump(final_model, 'f1_top3_pipeline.joblib')
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
