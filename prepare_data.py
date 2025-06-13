import pandas as pd
from fetch_f1_data import fetch_openf1_data, fetch_jolpica_data


def main():
    """Prepare minimal dataset with race and driver identifiers."""
    # Ensure raw data are available
    fetch_openf1_data(use_cache=True)
    fetch_jolpica_data(use_cache=True)

    results = pd.read_csv("jolpica_results.csv")
    races = pd.read_csv("jolpica_races.csv")

    results = results.rename(
        columns={
            "Driver.driverId": "driverId",
            "Constructor.constructorId": "constructorId",
        }
    )

    df = results.merge(
        races[["season", "round", "raceName"]],
        on=["season", "round", "raceName"],
        how="left",
    )

    df["race_id"] = df["season"] * 100 + df["round"]
    df["top3"] = pd.to_numeric(df["position"], errors="coerce") <= 3

    out = df[["race_id", "season", "driverId", "constructorId", "top3"]].copy()
    out.to_csv("processed_data.csv", index=False)
    print(f"processed_data.csv saved with {len(out)} rows")


if __name__ == "__main__":
    main()
