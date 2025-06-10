# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

st.title("F1 Top-3 Finish Predictie Dashboard")

# Feature preparation function
def prepare_features(df_sub):
    df_sub = df_sub.sort_values('date')
    df_sub['avg_finish_pos'] = df_sub.groupby('Driver.driverId')['finish_position'] \
                             .transform(lambda x: x.shift().expanding().mean())
    df_sub['avg_grid_pos']   = df_sub.groupby('Driver.driverId')['grid_position']   \
                             .transform(lambda x: x.shift().expanding().mean())
    df_sub['grid_diff']      = df_sub['avg_grid_pos'] - df_sub['grid_position']
    df_sub['driver_avg_Q3']  = df_sub.groupby('Driver.driverId')['Q3_sec']        \
                             .transform(lambda x: x.shift().expanding().mean())
    df_sub['Q3_diff']        = df_sub['driver_avg_Q3'] - df_sub['Q3_sec']
    # Removed weather and standings features
    return df_sub

# Load data and pipeline with caching
def load_data():
    return pd.read_csv('processed_data.csv', parse_dates=['date'])

def load_pipeline():
    return joblib.load('f1_top3_pipeline.joblib')

# Load once
df = load_data()
pipeline = load_pipeline()

# Sidebar for season and race selection
seasons = sorted(df['season'].unique())
selected_season = st.sidebar.selectbox('Selecteer seizoen', seasons, index=len(seasons)-1)

races = df[df['season']==selected_season]['raceName'].unique()
selected_race = st.sidebar.selectbox('Selecteer race', races)

selected_date = df[(df['season']==selected_season) & (df['raceName']==selected_race)]['date'].max()

# Split data for time-series evaluation
df_test = df[df['date'] == selected_date]

feature_cols = [
    'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
    'month', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
    'grid_diff', 'Q3_diff',
    'finish_rate_prev5',
    'team_qual_gap',

    'circuit_country', 'circuit_city',
        # Overtakes-features
    'weighted_overtakes',          # gewogen aantal inhaalacties
    'overtakes_per_lap',           # genormaliseerd per lap
    'weighted_overtakes_per_lap',   # gewogen én genormaliseerd
    'ewma_overtakes_per_lap',
    'ewma_weighted_overtakes_per_lap'
    ]



# Prepare test data
X_test = df_test[feature_cols]
y_test = df_test['top3']

# Predict probabilities and labels
proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

# Display Top-3 voorspellingen
df_test['top3_proba'] = proba

top3 = (
    df_test[['Driver.driverId','raceName','top3_proba']]
    .sort_values('top3_proba', ascending=False)
    .drop_duplicates('Driver.driverId')
    .head(3)
)

st.subheader(f"Top-3 voorspellingen voor {selected_race} {selected_season}")
st.table(top3.rename(columns={'Driver.driverId':'Coureur','top3_proba':'Kans'}))

# Performance metrics
st.subheader("Model Performance voor geselecteerde race")
st.write(f"ROC AUC: {roc_auc_score(y_test, proba):.3f}")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).T)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, proba)
fig1, ax1 = plt.subplots()
ax1.plot(fpr, tpr)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve')
st.pyplot(fig1)

# Precision-Recall curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, proba)
fig2, ax2 = plt.subplots()
ax2.plot(recall_vals, precision_vals)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
st.pyplot(fig2)

# --- Feature importance via permutation importance ---
st.subheader("Feature Importance")
perm = permutation_importance(
    pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
)
imp_df = (
    pd.DataFrame({"feature": feature_cols, "importance": perm.importances_mean})
    .sort_values("importance", ascending=False)
)
fig3, ax3 = plt.subplots()
ax3.barh(imp_df["feature"], imp_df["importance"])
ax3.set_xlabel("Permutation Importance")
ax3.invert_yaxis()
st.pyplot(fig3)

