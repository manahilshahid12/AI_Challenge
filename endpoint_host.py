# For deployment
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import datetime
import os 

app = Flask(__name__)

# Load the trained model
model = joblib.load(
    r"C:\Users\manah\OneDrive\Desktop\dps-challenge\AI_Challenge-1\model\rf_model.pkl"
)
# Load the dataset to calculate lag/rolling values
df = pd.read_csv(
    r"C:\Users\manah\Downloads\monatszahlen2505_verkehrsunfaelle_06_06_25.csv"
)
df = df[
    (df["MONATSZAHL"] == "Alkoholunfälle")
    & (df["AUSPRAEGUNG"] == "insgesamt")
    & (df["MONAT"].astype(str).str.lower() != "summe")
].copy()
df["MONAT"] = df["MONAT"].astype(str).str[-2:].astype(int)
df["date"] = pd.to_datetime(
    df["JAHR"].astype(str) + "-" + df["MONAT"].astype(str).str.zfill(2) + "-01"
)
df.set_index("date", inplace=True)
df.sort_index(inplace=True)
df["time_idx"] = np.arange(len(df))

"""""
@app.route("/", methods=["POST"])
def predict():
    data = request.get_json()
    year = int(data["year"])
    month = int(data["month"])

    try:
        pred_date = pd.Timestamp(f"{year}-{str(month).zfill(2)}-01")
        lags_needed = [1, 2, 3, 6, 12]
        lag_values = []
        for lag in lags_needed:
            lag_date = pred_date - pd.DateOffset(months=lag)
            lag_values.append(df.loc[lag_date, "WERT"])
        roll_3 = df.loc[
            pred_date - pd.DateOffset(months=3) : pred_date - pd.DateOffset(months=1),
            "WERT",
        ].mean()
        roll_6 = df.loc[
            pred_date - pd.DateOffset(months=6) : pred_date - pd.DateOffset(months=1),
            "WERT",
        ].mean()
        time_idx = df["time_idx"].max() + 1
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        X_pred = np.array(
            [[time_idx, month_sin, month_cos] + lag_values + [roll_3, roll_6]]
        )
        pred_log = model.predict(X_pred)[0]
        prediction = float(np.expm1(pred_log))
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    """


def make_prediction(year, month):
    pred_date = pd.Timestamp(f"{year}-{str(month).zfill(2)}-01")
    lags_needed = [1, 2, 3, 6, 12]
    lag_values = []
    for lag in lags_needed:
        lag_date = pred_date - pd.DateOffset(months=lag)
        lag_values.append(df.loc[lag_date, "WERT"])
    roll_3 = df.loc[
        pred_date - pd.DateOffset(months=3) : pred_date - pd.DateOffset(months=1),
        "WERT",
    ].mean()
    roll_6 = df.loc[
        pred_date - pd.DateOffset(months=6) : pred_date - pd.DateOffset(months=1),
        "WERT",
    ].mean()
    time_idx = df["time_idx"].max() + 1
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    X_pred = np.array(
        [[time_idx, month_sin, month_cos] + lag_values + [roll_3, roll_6]]
    )
    pred_log = model.predict(X_pred)[0]
    prediction = float(np.expm1(pred_log))
    return prediction


@app.route("/prediction", methods=["POST"])
def predict():
    data = request.get_json()
    year = int(data["year"])
    month = int(data["month"])
    try:
        prediction = make_prediction(year, month)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# running the app @ http://127.0.0.1:5000 (local host)
if __name__ == "__main__":
    app.run()
