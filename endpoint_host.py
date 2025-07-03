# For deployment
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import datetime

app = Flask(__name__)

# Load the trained model
model = joblib.load(
    r"C:\Users\manah\OneDrive\Desktop\dps-challenge\AI_Challenge-1\model\rf_model.pkl"
)
