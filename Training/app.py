# app.py
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# -----------------------------
# Load model and preprocessing
# -----------------------------
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # X_train.columns
categorical_cols = joblib.load("categorical_cols.pkl")
numeric_cols = joblib.load("numeric_cols.pkl")

print("Model and pipeline loaded.")
print("Expected features:", feature_columns)

# -----------------------------
# Helper: create date features
# -----------------------------
def add_date_features(df):
    df = df.copy()

    if "OCC_DATE" in df.columns:
        df["OCC_DATE"] = pd.to_datetime(df["OCC_DATE"], errors="coerce")
        df["OCC_YEAR"] = df["OCC_DATE"].dt.year
        df["OCC_MONTH"] = df["OCC_DATE"].dt.month
        df["OCC_DAY"] = df["OCC_DATE"].dt.day
        df["OCC_HOUR"] = df["OCC_DATE"].dt.hour
        df["OCC_DOW"] = df["OCC_DATE"].dt.dayofweek
        df["OCC_DOY"] = df["OCC_DATE"].dt.dayofyear

    if "REPORT_DATE" in df.columns:
        df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
        df["REPORT_MONTH"] = df["REPORT_DATE"].dt.month
        df["REPORT_DAY"] = df["REPORT_DATE"].dt.day
        df["REPORT_HOUR"] = df["REPORT_DATE"].dt.hour
        df["REPORT_DOW"] = df["REPORT_DATE"].dt.dayofweek
        df["REPORT_DOY"] = df["REPORT_DATE"].dt.dayofyear

    return df

# -----------------------------
# Prediction route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "No JSON payload received"}), 400

        # Convert JSON to DataFrame
        df = pd.DataFrame([json_data])

        # Add date-derived features
        df = add_date_features(df)

        # Fill missing features with NaN so pipeline imputes them
        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan

        # Reorder columns exactly as during training
        df = df[feature_columns]

        # Transform features using the pipeline
        X_prepared = pipeline.transform(df)

        # Make prediction
        prediction = model.predict(X_prepared)[0]

        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Health check
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Bike Theft Prediction API running"})

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=False)
    
#POSTMAN TEST JSON
#{
#   "OCC_DATE": "2013-12-26 05:00:00",
#   "REPORT_DATE": "2014-01-01 05:00:00",
#   "NEIGHBOURHOOD_158": "77",
#   "BIKE_MODEL": "Trek",
#   "BIKE_TYPE": "RC",
#   "PRIMARY_OFFENCE": "B&E",
#   "BIKE_COST": 1300,
#   "PREMISES_TYPE": "Commercial",
#   "LOCATION_TYPE": "Commercial",
#   "EVENT_UNIQUE_ID": "GO-20141263544",
#   "BIKE_SPEED": 6,
#   "NEIGHBOURHOOD_140": "165",
#   "BIKE_COLOUR": "SILRED"
# }
