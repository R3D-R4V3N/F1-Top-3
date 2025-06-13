# F1-Forecast (Minimal)

This simplified project predicts whether a driver will finish in the top three of a Formula&nbsp;1 race.  
Raw data is downloaded from the public OpenF1 and Jolpica APIs and cached locally.  
Only a handful of identifiers are kept as features:
`race_id`, `season`, `driverId` and `constructorId`.

## Workflow

1. **Fetch data**
   ```bash
   python fetch_f1_data.py
   ```
   The script downloads all required JSON endpoints and stores them as CSV files.  
   Requests are skipped when the files already exist so repeated runs are fast.

2. **Prepare dataset**
   ```bash
   python prepare_data.py
   ```
   Creates `processed_data.csv` with the columns above plus a `top3` label.

3. **Train model**
   ```bash
   python train_model_catboost.py
   ```
   Trains a basic CatBoost classifier and writes its ROC&ndash;AUC to
   `model_performance/catboost_model_performance.csv`.

4. **Export model**
   ```bash
   python export_model.py
   ```
   Fits the final model on the full dataset and saves `f1_top3_pipeline.joblib`.

5. **Inference**
   ```bash
   python infer.py
   ```
   Example script that prints the top three probabilities for a given `race_id`.

6. **Streamlit dashboard**
   ```bash
   streamlit run streamlit_app.py
   ```
   Select a season and race to view the model's top‑three predictions.

## Requirements

Install the dependencies with:
```bash
pip install -r requirements.txt
```
