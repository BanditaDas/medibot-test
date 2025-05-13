from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Added for CORS support
import joblib
import pandas as pd
import re
from fuzzywuzzy import process
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://final-year-project-t7pw.onrender.com"}})  # Or your actual Node.js domain


# Load model and data
model = joblib.load("model.pkl")  # Ensure model.pkl is in the same directory as app.py
df = pd.read_csv("diseases.csv")  # Ensure diseases.csv is in the same directory


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
    
    # Convert the symptoms into the appropriate format for the model
    # Assuming the model takes a list of symptom strings as input (adjust accordingly)
    probabilities = model.predict_proba([symptoms])[0]
    
    # Sort the probabilities and return top_n predictions
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    return model.classes_[top_indices].tolist()


# Home route to serve HTML page
@app.route("/")
def home():
    return render_template("index.html")  # Ensure you have an 'index.html' file in the templates directory


# API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_input = data.get("text", "")
    cleaned_symptoms = extract_symptoms(user_input, symptom_list)  # Extract symptoms using FuzzyWuzzy
    predictions = predict_diseases(model, cleaned_symptoms)  # Get top predictions from the model
    return jsonify({"predictions": predictions})


# Health check route (useful for testing if the app is running)
@app.route('/ping', methods=['GET'])
def ping():
    return "pong", 200


if __name__ == "__main__":
    app.run(debug=True)  # Run the app in debug mode
