import argparse
import joblib
import pandas as pd
from sklearn.base import clone

from train_model_catboost import build_and_train_pipeline


def save_pipeline():
    """Train and save the CatBoost pipeline on the full dataset."""
    model, _ = build_and_train_pipeline()

    df = pd.read_csv("processed_data.csv")
    X = df[["race_id", "season", "driverId", "constructorId"]]
    y = df["top3"]

    final_model = clone(model)
    final_model.fit(X, y, cat_features=[2, 3])

    joblib.dump(final_model, "f1_top3_pipeline.joblib")
    print("Pipeline saved to f1_top3_pipeline.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained pipeline")
    _ = parser.parse_args()
    save_pipeline()
