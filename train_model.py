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
