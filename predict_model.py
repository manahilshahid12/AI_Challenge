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
    # Time index
    time_idx = len(df)

    # Create feature vector
    X_pred = np.array(
        [[time_idx, month_sin, month_cos] + lag_values + [roll_3, roll_6]]
    )

    # Predict
    pred_log = model.predict(X_pred)[0]
    pred = np.expm1(pred_log)
    # print(
    #   f"\n Predicted number of alcohol-related accidents for {year}-{str(month).zfill(2)}: {pred:.2f}"
    # )
    print(
        f"\n Category: Alkoholunfälle, \n Type: Insgesamt, \n Year: 2021, \n Month: 01",
        f"\n Value= {pred:.2f}",
    )

    # Compare with actual values for year 2021.01.01
    if target_date in df.index:
        actual = df.loc[target_date, "WERT"]
        error = abs(actual - pred)
        print(f" Actual value: {actual}")
        print(f" Absolute error: {error:.2f}")
    else:
        print("Actual value not available in dataset.")


# Forecasting Values for
"""''
Category: 'Alkoholunfälle'
Type: 'insgesamt
Year: '2021'
Month: '01'
"""
if __name__ == "__main__":
    predict_value(
        category="Alkoholunfälle",
        value_type="insgesamt",
        year=2021,
        month=1,
        model_path="model/rf_model.pkl",
        data_path=r"C:\Users\manah\OneDrive\Desktop\dps-challenge\AI_Challenge-1\monatszahlen2505_verkehrsunfaelle_06_06_25.csv",
    )
