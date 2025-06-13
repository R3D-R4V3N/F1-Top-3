import joblib
import pandas as pd


def predict_for_race(race_id: int) -> None:
    """Predict top-3 finish probabilities for a given race."""
    df = pd.read_csv("processed_data.csv")
    df_race = df[df["race_id"] == race_id]
    if df_race.empty:
        print(f"Race id {race_id} not found in processed data")
        return

    model = joblib.load("f1_top3_pipeline.joblib")
    X = df_race[["race_id", "season", "driverId", "constructorId"]]
    proba = model.predict_proba(X)[:, 1]
    df_race = df_race.assign(top3_proba=proba)
    top3 = (
        df_race.sort_values("top3_proba", ascending=False)
        .drop_duplicates("driverId")
        .head(3)
    )
    print(top3[["driverId", "top3_proba"]])


if __name__ == "__main__":
    predict_for_race(202201)  # example: 2022 season round 1 -> race_id 202201
