import os
import pandas as pd
import gspread
import json
import base64
from google.oauth2.service_account import Credentials
from flask import Flask, jsonify
from transformers import pipeline
from rapidfuzz import process, fuzz
import re

app = Flask(__name__)

# 🔹 Load Google Credentials (Dukungan Base64)
encoded_creds = os.getenv("GOOGLE_SERVICE_ACCOUNT_BASE64")
if encoded_creds:
    decoded_creds = base64.b64decode(encoded_creds).decode()
    google_json = json.loads(decoded_creds)
else:
    raise ValueError("GOOGLE_SERVICE_ACCOUNT_BASE64 tidak tersedia di environment variables")

creds = Credentials.from_service_account_info(google_json, scopes=["https://www.googleapis.com/auth/spreadsheets"])
client = gspread.authorize(creds)

# 🔹 Load Google Sheet
SHEET_URL = os.getenv("GOOGLE_SHEET_URL")
if not SHEET_URL:
    raise ValueError("GOOGLE_SHEET_URL tidak tersedia di environment variables")

try:
    sheet = client.open_by_url(SHEET_URL)
    worksheet = sheet.sheet1
except Exception as e:
    raise ValueError(f"Error membuka Google Sheet: {str(e)}")

# 🔹 Load IndoBERT Sentiment Model
sentiment_model = pipeline("text-classification", model="w11wo/indonesian-roberta-base-sentiment-classifier")

# 🔹 Daftar Keyword dan Entitas
NEGATIVE_KEYWORDS = ["mati", "sakit", "lemas", "muntah", "menggigil", "kurus", "kenapa", "tolong", "meninggal", "terkapar", "demam", "lesu", "pingsan", "tidak mau makan", "drop", "lemes"]
POSITIVE_KEYWORDS = ["sehat", "baik", "aman", "damai", "bagus", "stabil", "tidak apaapa"]
RELEVANT_ENTITIES = ["sapi", "kerbau", "ternak", "ayam", "domba", "bebek", "kambing", "itik", "peternakan", "hewan ternak"]
UNRELATED_PHRASES = ["hari ini panas", "belum makan siang", "tidak ada makanan enak"]

VOCAB = NEGATIVE_KEYWORDS + POSITIVE_KEYWORDS + RELEVANT_ENTITIES

# 🔹 Fungsi Ambil Data dari Google Sheets
def get_data_from_sheets():
    try:
        data = worksheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        raise RuntimeError(f"Error mengambil data dari Google Sheets: {str(e)}")

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
    try:
        df = get_data_from_sheets()
        if df.empty:
            return jsonify({"error": "Tidak ada data dalam Google Sheets"}), 400

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

        # Format tanggal sebelum update ke Google Sheets
        df["tanggal"] = df["tanggal"].astype(str)
        daily_sentiment["tanggal"] = daily_sentiment["tanggal"].dt.strftime("%Y-%m-%d")

        print("Data agregasi sentimen per hari:")
        print(daily_sentiment.head())

        # Update data utama ke Google Sheets
        worksheet.update("A1", [df.columns.tolist()] + df.values.tolist())
        print("Data utama diperbarui di Google Sheets.")

        # Update data agregasi ke worksheet "daily_sentiment"
        daily_worksheet = sheet.worksheet("daily_sentiment")
        daily_worksheet.update("A1", [daily_sentiment.columns.tolist()] + daily_sentiment.values.tolist())
        print("Data agregasi diperbarui di Google Sheets.")

        return jsonify({"message": "Sentimen & agregasi berhasil diperbarui", "total": len(df)})

    except Exception as e:
        print(f"Error dalam proses update: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 🔹 Endpoint API untuk Cek Status
@app.route("/")
def home():
    return jsonify({"message": "API is running"}), 200

# 🔹 Menjalankan Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
