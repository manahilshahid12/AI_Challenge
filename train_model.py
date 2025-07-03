# train_model.py
# Import the libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
import os

# Load and filter dataset as per instructed for year, category and type of accidents
df = pd.read_csv(
    r"C:\Users\manah\Downloads\monatszahlen2505_verkehrsunfaelle_06_06_25.csv"
)
df = df[
    (df["JAHR"] <= 2020)
    & (df["MONATSZAHL"] == "Alkoholunfälle")
    & (df["AUSPRAEGUNG"] == "insgesamt")
    & (df["MONAT"].astype(str).str.lower() != "summe")
].copy()

# Time handling
# Creating YY-MM-DD format for month sice values are not interpretable
df["MONAT"] = df["MONAT"].astype(str).str[-2:].astype(int)
df["date"] = pd.to_datetime(
    df["JAHR"].astype(str) + "-" + df["MONAT"].astype(str).str.zfill(2) + "-01"
)
df.sort_values("date", inplace=True)
df.set_index("date", inplace=True)

# To make model more adaptive and robust
# Adding features
# numeric time index for progression
df["time_idx"] = np.arange(len(df))

# Adding cyclic features to timeseries data to capture the cyclic trend
df["month_sin"] = np.sin(2 * np.pi * df["MONAT"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["MONAT"] / 12)

# Add lag features to compare previous months value and rolling means
# Average of total values of 3,6, 9 months
for lag in [1, 2, 3, 6, 12]:
    df[f"lag_{lag}"] = df["WERT"].shift(lag)
df["roll_3"] = df["WERT"].rolling(window=3).mean().shift(1)
df["roll_6"] = df["WERT"].rolling(window=6).mean().shift(1)
# df['roll_9'] = df['WERT'].rolling(window=9).mean().shift(1)

# Target transformation: log1p to reduce skewness
# unexpected values
df["WERT_log"] = np.log1p(df["WERT"])

# Drop rows with NaN (due to lags/rolling)
df.dropna(inplace=True)

# For more features enable roll 9 and add here (in case of extended dataset/causes redundancy)
# Creating a feature space for training
# Features for training
features = [
    "time_idx",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "roll_3",
    "roll_6",
]
# Features for training
X = df[features].values
y = df["WERT_log"].values  # Using transformed target

# Train/test split could be conventional 80/20
"""""
X_train, y_train, X_test, y_test = train_test_split(
    X, Y, random_state=42, test_size=0.2
"""
# or train before 2020 and test on 2020 to visualize the error
# Train/test split: train before 2020, test in 2020
train_mask = df["JAHR"] < 2020
test_mask = df["JAHR"] == 2020
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# TimeSeriesSplit for cross-validation tuning
tscv = TimeSeriesSplit(n_splits=5)

# Parameter tuning for hyperparameter tunning of the regression model
# Search space defined
param_grid = {
    "n_estimators": [100, 300, 500, 700],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5, 7],
    "min_samples_leaf": [1, 2, 5],
}
# Initiating the model
rf = RandomForestRegressor(random_state=42)

# Gridsearch for best hyperparameters using tscv
grid_search = GridSearchCV(
    rf, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

# Using the best model
best_rf = grid_search.best_estimator_

# Predict on test (log scale)
y_pred_log = best_rf.predict(X_test)
y_pred = np.expm1(y_pred_log)  # inverse transform
y_true = np.expm1(y_test)

mse = mean_squared_error(y_true, y_pred) / 10
print(f"Test MSE (original scale): {mse:.2f}")

"""""
# Plot actual vs predicted for test year 2020
#Optional to see the Actual vs predicted on test
plt.figure(figsize=(10,5))
plt.plot(df.loc[test_mask].index, y_true, label='Actual')
plt.plot(df.loc[test_mask].index, y_pred, label='Predicted')
plt.title('Alcohol-related accidents: Actual vs Predicted (2020)')
plt.xlabel('Date')
plt.ylabel('Number of Accidents')
plt.legend()
plt.show()
"""
MODEL_PATH = "model/rf_model.pkl"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(best_rf, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")
