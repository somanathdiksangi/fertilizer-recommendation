from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS
import pickle
import pandas as pd
import numpy as np
import logging

from preprocess import preprocess_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

fertilizer_mapping = {
    1: "Urea", 2: "DAP", 3: "MOP", 4: "10:26:26 NPK", 5: "SSP",
    6: "Magnesium Sulphate", 7: "13:32:26 NPK", 8: "12:32:16 NPK",
    9: "50:26:26 NPK", 10: "19:19:19 NPK", 11: "Chilated Micronutrient",
    12: "18:46:00 NPK", 13: "Sulphur", 14: "20:20:20 NPK",
    15: "Ammonium Sulphate", 16: "Ferrous Sulphate", 17: "White Potash",
    18: "10:10:10 NPK", 19: "Hydrated Lime"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.content_type == "application/json":
            data = request.get_json()
        else:
            data = request.form.to_dict()

        logging.info(f"Received Data: {data}")

        required_fields = ["district", "soil_color", "nitrogen", "phosphorus", "potassium", "pH", "rainfall", "temperature", "crop"]
        missing_fields = [field for field in required_fields if field not in data or data[field] == ""]

        if missing_fields:
            logging.error(f"Missing fields: {missing_fields}")
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        try:
            processed_data = {
                "District_Name": data["district"],
                "Soil_color": data["soil_color"],
                "Nitrogen": float(data["nitrogen"]),
                "Phosphorus": float(data["phosphorus"]),
                "Potassium": float(data["potassium"]),
                "pH": float(data["pH"]),
                "Rainfall": float(data["rainfall"]),
                "Temperature": float(data["temperature"]),
                "Crop": data["crop"]
            }
        except ValueError as ve:
            logging.error(f"Invalid input type: {ve}")
            return jsonify({"error": "Invalid input: Ensure all numerical values are valid."}), 400

        if model is None:
            logging.error("Model is not loaded.")
            return jsonify({"error": "Model is not available."}), 500

        df = preprocess_input(processed_data)
        prediction = model.predict(df)[0]

        if isinstance(prediction, np.generic):
            prediction = prediction.item()

        logging.info(f"Raw Model Prediction: {prediction}")

        fertilizer_name = fertilizer_mapping.get(prediction, "Unknown Fertilizer")
        logging.info(f"Mapped Fertilizer: {fertilizer_name}")

        return jsonify({"recommended_fertilizer": fertilizer_name})

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)