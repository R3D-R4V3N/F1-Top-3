import streamlit as st
import pandas as pd
import joblib

st.title("F1 Top-3 Prediction")


@st.cache_data
def load_data():
    return pd.read_csv("processed_data.csv")


@st.cache_resource
def load_model():
    return joblib.load("f1_top3_pipeline.joblib")


df = load_data()
model = load_model()

season = st.selectbox("Season", sorted(df["season"].unique()))
available_races = sorted(df[df["season"] == season]["race_id"].unique())
race_id = st.selectbox("Race", available_races)

race_rows = df[df["race_id"] == race_id]
X = race_rows[["race_id", "season", "driverId", "constructorId"]]
proba = model.predict_proba(X)[:, 1]
race_rows = race_rows.assign(probability=proba)
top3 = race_rows.sort_values("probability", ascending=False).drop_duplicates("driverId").head(3)

st.subheader("Top-3 prediction")
st.table(top3[["driverId", "probability"]])
