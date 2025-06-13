import os
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score


def build_and_train_pipeline(export_csv: bool = True,
                             csv_path: str = "model_performance/catboost_model_performance.csv"):
    """Train a simple CatBoost model on minimal features."""
    df = pd.read_csv("processed_data.csv")
    features = ["race_id", "season", "driverId", "constructorId"]
    X = df[features]
    y = df["top3"]

    unique_races = df["race_id"].drop_duplicates().sort_values()
    split = int(len(unique_races) * 0.8)
    train_races = unique_races.iloc[:split]
    test_races = unique_races.iloc[split:]

    train_mask = df["race_id"].isin(train_races)
    test_mask = df["race_id"].isin(test_races)

    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        verbose=0,
        random_state=42,
    )

    model.fit(X[train_mask], y[train_mask], cat_features=[2, 3])

    proba = model.predict_proba(X[test_mask])[:, 1]
    preds = model.predict(X[test_mask])
    auc = roc_auc_score(y[test_mask], proba)
    report = classification_report(y[test_mask], preds)
    print(report)
    print(f"Test ROC AUC: {auc:.3f}")

    if export_csv:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame({"Metric": ["Test ROC AUC"], "Value": [auc]}).set_index("Metric").to_csv(csv_path)

    return model, None


def main():
    build_and_train_pipeline()


if __name__ == "__main__":
    main()
