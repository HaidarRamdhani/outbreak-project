import os
import pandas as pd
import gspread
import json
from google.oauth2.service_account import Credentials
from flask import Flask, request, jsonify
from transformers import pipeline
from rapidfuzz import process, fuzz
import re

app = Flask(__name__)

# Load kredensial dari Railway Environment
google_json = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
creds = Credentials.from_service_account_info(google_json, scopes=["https://www.googleapis.com/auth/spreadsheets"])
client = gspread.authorize(creds)

# Buka Google Sheets
SHEET_URL = os.getenv("GOOGLE_SHEETS_URL")
sheet = client.open_by_url(SHEET_URL)
worksheet = sheet.sheet1

# Load Model Sentimen IndoBERT
sentiment_model = pipeline("text-classification", model="w11wo/indonesian-roberta-base-sentiment-classifier")

def preprocess_text(text):
    text = text.strip().lower()
    text = re.sub(r"(.)\1{2,}", r"\1", text)  # Hilangkan huruf berulang
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)  # Hilangkan karakter non-alfanumerik
    return text

def classify_sentiment(text):
    result = sentiment_model(text)[0]
    return 1 if result['label'] == "NEGATIVE" and result['score'] > 0.15 else 0

@app.route("/update_sentiment", methods=["POST"])
def update_sentiment():
    df = pd.DataFrame(worksheet.get_all_records())
    df["preprocessed_text"] = df["jawaban"].apply(preprocess_text)
    df["sentimen_negatif"] = df["preprocessed_text"].apply(classify_sentiment)
    
    # Update ke Google Sheets
    worksheet.update("A1", [df.columns.tolist()] + df.values.tolist())
    return jsonify({"message": "Sentimen berhasil diperbarui", "total": len(df)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
