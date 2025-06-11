# export_model.py

import argparse
import joblib
from sklearn.base import clone
import pandas as pd

from train_model_catboost import build_and_train_pipeline as build_catb


def save_pipeline():
    """Train en bewaar het CatBoost-model."""

    pipeline, best_params = build_catb()

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
    args = parser.parse_args([])
    save_pipeline()
