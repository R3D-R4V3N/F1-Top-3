# train_model_lgbm_ranker.py
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score


def train_lgbm_ranker(export_csv: bool = True,
                      csv_path: str = "model_performance/lgbm_ranker_performance.csv"):
    """Train a LightGBM ranking model and optionally export performance metrics.

    Parameters
    ----------
    export_csv : bool, optional
        Whether to write metrics to ``csv_path``. Defaults to ``True``.
    csv_path : str, optional
        File where metrics will be stored. Defaults to
        ``model_performance/lgbm_ranker_performance.csv``.
    """

    # 1. Load and prepare data
    df = pd.read_csv("processed_data.csv", parse_dates=["date"])
    df = df.sort_values("date")
    df["race_id"] = df["season"] * 100 + df["round"]

    # Drop rows without finish position
    df = df.dropna(subset=["finish_position"])

    # Relevance label for ranking: higher is better
    df["relevance"] = 1 / df["finish_position"]

    numeric_feats = [
        "grid_position", "Q1_sec", "Q2_sec", "Q3_sec",
        "month", "avg_finish_pos", "avg_grid_pos", "avg_const_finish",
        "grid_diff", "Q3_diff",
        "finish_rate_prev5",
        "team_qual_gap",
        # Overtake related features
        "weighted_overtakes",
        "overtakes_per_lap",
        "weighted_overtakes_per_lap",
        "ewma_overtakes_per_lap",
        "ewma_weighted_overtakes_per_lap",
    ]
    categorical_feats = ["circuit_country", "circuit_city"]

    X = df[numeric_feats + categorical_feats]
    y = df["relevance"]

    # 2. Train/test split based on entire races
    unique_races = df["race_id"].drop_duplicates()
    split_idx = int(len(unique_races) * 0.8)
    train_races = unique_races.iloc[:split_idx]
    test_races = unique_races.iloc[split_idx:]

    train_mask = df["race_id"].isin(train_races)
    test_mask = df["race_id"].isin(test_races)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    train_group = df[train_mask].groupby("race_id").size().tolist()
    test_group = df[test_mask].groupby("race_id").size().tolist()

    # 3. Preprocessing pipelines
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_feats),
        ("cat", cat_pipe, categorical_feats),
    ])

    # 4. Ranking model
    ranker = LGBMRanker(
        objective="lambdarank",
        random_state=42,
        learning_rate=0.05,
        n_estimators=500,
    )

    model = Pipeline([
        ("pre", pre),
        ("ranker", ranker),
    ])

    model.fit(
        X_train,
        y_train,
        ranker__group=train_group,
        ranker__eval_set=[(X_test, y_test)],
        ranker__eval_group=[test_group],
        ranker__eval_at=[10],
        ranker__early_stopping_rounds=50,
    )

    # 5. Evaluate NDCG@10 on test races
    scores = []
    test_df = df[test_mask]
    for rid, grp in test_df.groupby("race_id"):
        X_grp = grp[numeric_feats + categorical_feats]
        y_true = grp["relevance"].values.reshape(1, -1)
        y_pred = model.predict(X_grp).reshape(1, -1)
        scores.append(ndcg_score(y_true, y_pred, k=10))
    test_ndcg = float(np.mean(scores))

    print(f"Mean NDCG@10 on test races: {test_ndcg:.4f}")

    if export_csv:
        perf_df = pd.DataFrame(
            {"Metric": ["Test NDCG@10"], "Value": [test_ndcg]}
        ).set_index("Metric")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        perf_df.to_csv(csv_path)
        print(f"Performance metrics saved to {csv_path}")

    return model


def main() -> None:
    train_lgbm_ranker()


if __name__ == "__main__":
    main()
