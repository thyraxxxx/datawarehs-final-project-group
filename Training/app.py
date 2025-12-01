# app.py
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# -----------------------------
# Load model and preprocessing
# -----------------------------
try:
    model = joblib.load("model.pkl")
    pipeline = joblib.load("pipeline.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    categorical_cols = joblib.load("categorical_cols.pkl")
    numeric_cols = joblib.load("numeric_cols.pkl")
    print("Model and pipeline loaded successfully!")
    print("Expected features:", feature_columns)
except Exception as e:
    print(f" Error loading model files: {e}")

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
# Route to serve HTML
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Health check
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "expected_features": len(feature_columns)
    })

# -----------------------------
# Prediction route (ONLY ONE!)
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "No JSON payload received"}), 400
        
        print("Received data:", json_data)
        
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
        
        print("DataFrame shape:", df.shape)
        
        # Transform features using the pipeline
        X_prepared = pipeline.transform(df)
        
        print("Transform successful. Shape:", X_prepared.shape)
        
        # Make prediction
        prediction = model.predict(X_prepared)
        print("Raw prediction:", prediction, "Type:", type(prediction))
        
        # Handle both text and numeric predictions
        if isinstance(prediction, np.ndarray):
            pred_value = prediction[0]
        else:
            pred_value = prediction
        
        # Convert text predictions to binary
        if isinstance(pred_value, str):
            pred_upper = str(pred_value).upper()
            # Check if bike was recovered/returned
            if any(word in pred_upper for word in ['RECOVER', 'RETURN', 'FOUND']):
                prediction_value = 1  # Returned
            else:
                prediction_value = 0  # Not returned (STOLEN, UNKNOWN, etc.)
            print(f"Converted '{pred_value}' to {prediction_value}")
        else:
            # Already numeric
            prediction_value = int(pred_value)
        
        # Get probability (confidence)
        try:
            probabilities = model.predict_proba(X_prepared)
            print("Raw probabilities:", probabilities)
            
            if isinstance(probabilities, np.ndarray) and probabilities.ndim == 2:
                prob_array = probabilities[0]
                # Get max probability as confidence
                confidence = float(max(prob_array) * 100)
            else:
                confidence = 75.0
        except Exception as prob_error:
            print(f"Could not get probability: {prob_error}")
            confidence = 75.0
        
        print(f"Final - Prediction: {prediction_value}, Confidence: {confidence:.1f}%")
        
        return jsonify({
            "prediction": prediction_value,
            "probability": confidence
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser")
    print("Press CTRL+C to stop\n")
    app.run(debug=True, port=5000)
    
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
