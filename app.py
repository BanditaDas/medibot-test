from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Added for CORS support
import joblib
import pandas as pd
import re
from fuzzywuzzy import process
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:8080"}})  # Or your actual Node.js domain


# Load model and data
model = joblib.load("model.pkl")
df = pd.read_csv("diseases_symptoms_medicines.csv")

# Generate symptom list
def get_symptom_list(df):
    symptoms = set()
    for entry in df["Symptoms"]:
        symptoms.update([sym.strip().lower() for sym in entry.split(",")])
    return list(symptoms)

symptom_list = get_symptom_list(df)

# Fuzzy symptom extractor
def extract_symptoms(text, symptom_list):
    words = re.findall(r'\b\w+\b', text.lower())
    matched_symptoms = set()
    for word in words:
        match, score = process.extractOne(word, symptom_list)
        if score > 80:
            matched_symptoms.add(match)
    return ",".join(matched_symptoms)

# Prediction logic
def predict_diseases(model, symptoms, top_n=3):
    if not symptoms:
        return ["No relevant symptoms detected."]
    probabilities = model.predict_proba([symptoms])[0]
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    return model.classes_[top_indices].tolist()

# Home route to serve HTML page
@app.route("/")
def home():
    return render_template("index.html")

# API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_input = data.get("text", "")
    cleaned_symptoms = extract_symptoms(user_input, symptom_list)
    predictions = predict_diseases(model, cleaned_symptoms)
    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(debug=True)
