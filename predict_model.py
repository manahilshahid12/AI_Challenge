import pandas as pd
import numpy as np
import joblib


# Define a prediction function for predicting the number of accidents
def predict_value(category, value_type, year, month, model_path, data_path):
    # Load trained model
    model = joblib.load(model_path)

    # Load raw data for lag and rolling calculations
    df = pd.read_csv(data_path)
    df = df[
        (df["MONATSZAHL"] == category)
        & (df["AUSPRAEGUNG"] == value_type)
        & (df["MONAT"].astype(str).str.lower() != "summe")
    ].copy()

    df["MONAT"] = df["MONAT"].astype(str).str[-2:].astype(int)
    df["date"] = pd.to_datetime(
        df["JAHR"].astype(str) + "-" + df["MONAT"].astype(str).str.zfill(2) + "-01"
    )
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # Build date for prediction
    try:
        target_date = pd.to_datetime(f"{year}-{str(month).zfill(2)}-01")
    except Exception as e:
        print(f"Invalid date: {e}")
        return

    # Cyclical features
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Lags
    lag_values = []
    for lag in [1, 2, 3, 6, 12]:
        lag_date = target_date - pd.DateOffset(months=lag)
        if lag_date not in df.index:
            print(f"Missing data for lag date: {lag_date}")
            return
        lag_values.append(df.loc[lag_date, "WERT"])

    # Rolling means
    try:
        roll_3 = df.loc[
            target_date
            - pd.DateOffset(months=3) : target_date
            - pd.DateOffset(months=1),
            "WERT",
        ].mean()
        roll_6 = df.loc[
            target_date
            - pd.DateOffset(months=6) : target_date
            - pd.DateOffset(months=1),
            "WERT",
        ].mean()
    except Exception as e:
        print(f"Insufficient data for rolling features: {e}")
        return
