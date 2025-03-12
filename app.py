import os
import pandas as pd
import gspread
import json
from google.oauth2.service_account import Credentials
from flask import Flask, jsonify
from transformers import pipeline
from rapidfuzz import process, fuzz
import re

app = Flask(__name__)

# 🔹 Load kredensial dari Railway Environment
google_json = json.load(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
creds = Credentials.from_service_account_info(google_json, scopes=["https://www.googleapis.com/auth/spreadsheets"])
client = gspread.authorize(creds)

# 🔹 Buka Google Sheets
SHEET_URL = os.getenv("GOOGLE_SHEET_URL")  # Ambil dari Railway
sheet = client.open_by_url(SHEET_URL)
worksheet = sheet.sheet1

# 🔹 Load IndoBERT sentiment model
sentiment_model = pipeline("text-classification", model="w11wo/indonesian-roberta-base-sentiment-classifier")

# 🔹 Daftar keyword negatif dan entitas terkait ternak
NEGATIVE_KEYWORDS = ["mati", "sakit", "lemas", "muntah", "menggigil", "kurus", "kenapa", "tolong", "meninggal", "terkapar", "demam", "lesu", "pingsan", "tidak mau makan", "drop", "lemes"]
POSITIVE_KEYWORDS = ["sehat", "baik", "aman", "damai", "bagus", "stabil", "tidak apaapa"]
RELEVANT_ENTITIES = ["sapi", "kerbau", "ternak", "ayam", "domba", "bebek", "kambing", "itik", "peternakan", "hewan ternak"]
UNRELATED_PHRASES = ["hari ini panas", "belum makan siang", "tidak ada makanan enak"]

VOCAB = NEGATIVE_KEYWORDS + POSITIVE_KEYWORDS + RELEVANT_ENTITIES

# 🔹 Fungsi Ambil Data dari Google Sheets
def get_data_from_sheets():
    data = worksheet.get_all_records()
    return pd.DataFrame(data)

# 🔹 Fungsi Koreksi Typo dengan RapidFuzz
def correct_typo(text, vocab, threshold=85):
    corrected_words = []
    for word in text.split():
        match = process.extractOne(word, vocab, scorer=fuzz.partial_ratio)
        if match and match[1] >= threshold:
            corrected_words.append(match[0])
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

# 🔹 Fungsi Normalisasi Teks
def normalize_text(text):
    text = re.sub(r"(.)\1{2,}", r"\1", text)  # Hilangkan huruf berulang
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)  # Hilangkan karakter non-alfanumerik
    return text

# 🔹 Fungsi Preprocessing
def preprocess_text(text):
    text = text.strip().lower()
    text = normalize_text(text)
    text = correct_typo(text, VOCAB)
    return text

# 🔹 Fungsi Rule-Based Sentimen
def rule_based_sentiment(text):
    if any(unrelated in text for unrelated in UNRELATED_PHRASES):
        return 0  # Tidak relevan = Netral

    if any(pos in text for pos in POSITIVE_KEYWORDS):
        return 0  # Positif = Netral

    if any(entity in text for entity in RELEVANT_ENTITIES) and any(word in text for word in NEGATIVE_KEYWORDS):
        return 1  # Negatif jika terkait ternak

    return None  # Tidak bisa ditentukan, lanjut ke IndoBERT

# 🔹 Fungsi Klasifikasi Sentimen
def classify_sentiment(text):
    rule_result = rule_based_sentiment(text)
    if rule_result is not None:
        return rule_result
    result = sentiment_model(text)[0]
    return 1 if result['label'] == "NEGATIVE" and result['score'] > 0.15 else 0

# 🔹 Endpoint API untuk Update Sentimen
@app.route("/update_sentiment", methods=["POST"])
def update_sheets_with_sentiment():
    df = get_data_from_sheets()
    
    print("Data awal dari Google Sheets:")
    print(df.head())

    df["preprocessed_text"] = df["jawaban"].apply(preprocess_text)
    df["sentimen_negatif"] = df["preprocessed_text"].apply(classify_sentiment)

    print("Data setelah klasifikasi sentimen:")
    print(df[["jawaban", "sentimen_negatif"]].head())

    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")

    print("Baris dengan NaT setelah parsing:")
    print(df[df["tanggal"].isna()])

    daily_sentiment = df.groupby("tanggal")["sentimen_negatif"].sum().reset_index()
    daily_sentiment = daily_sentiment.dropna(subset=["tanggal"])
    df["tanggal"] = df["tanggal"].astype(str)
    daily_sentiment["tanggal"] = daily_sentiment["tanggal"].dt.strftime("%Y-%m-%d")

    print("Data agregasi sentimen per hari:")
    print(daily_sentiment.head())

    worksheet.update("A1", [df.columns.tolist()] + df.values.tolist())
    print("Data utama diperbarui di Google Sheets.")

    daily_worksheet = sheet.worksheet("daily_sentiment")
    daily_worksheet.update("A1", [daily_sentiment.columns.tolist()] + daily_sentiment.values.tolist())
    print("Data agregasi diperbarui di Google Sheets.")

    return jsonify({"message": "Sentimen & agregasi berhasil diperbarui", "total": len(df)})

# 🔹 Endpoint API untuk Cek Status
@app.route("/")
def home():
    return jsonify({"message": "API is running"}), 200

# 🔹 Menjalankan Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
