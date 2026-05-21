
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from newspaper import Article
import requests
import os

app = Flask(__name__)

# Proper CORS setup
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------------
# LOAD MODEL (FIXED PATH)
# -------------------------
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# -------------------------
# TRUSTED SOURCES
# -------------------------
TRUSTED_SOURCES = [
    "bbc.com",
    "reuters.com",
    "cnn.com",
    "nytimes.com",
    "theguardian.com",
    "aljazeera.com"
]

def check_source_credibility(url):
    for source in TRUSTED_SOURCES:
        if source in url.lower():
            return "HIGH"
    return "UNKNOWN"

@app.route("/")
def home():
    return "Fake News Detection API Running"

# -------------------------
# ML Prediction Core
# -------------------------
def analyze_text(text):
    text = text.lower()[:3000]

    vect = vectorizer.transform([text])

    prediction = model.predict(vect)[0]
    probs = model.predict_proba(vect)[0]

    real_prob = float(probs[0])
    fake_prob = float(probs[1])

    result = "FAKE" if prediction == 1 else "REAL"
    confidence = fake_prob if result == "FAKE" else real_prob

    return result, confidence

# -------------------------
# TEXT INPUT
# -------------------------
@app.route("/predict-text", methods=["POST"])
def predict_text():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]

        result, confidence = analyze_text(text)

        return jsonify({
            "prediction": result,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# URL INPUT
# -------------------------
@app.route("/predict-url", methods=["POST"])
def predict_url():
    try:
        data = request.get_json()

        if not data or "url" not in data:
            return jsonify({"error": "No URL provided"}), 400

        url = data["url"]

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=5)

        article = Article(url)
        article.set_html(response.text)
        article.parse()

        article_text = article.title + " " + article.text

        if len(article_text) < 50:
            return jsonify({"error": "Article content extraction failed"}), 400

        result, confidence = analyze_text(article_text)

        credibility = check_source_credibility(url)

        final_prediction = result

        if credibility == "HIGH" and result == "FAKE":
            final_prediction = "LIKELY REAL"

        return jsonify({
            "prediction": result,
            "final_prediction": final_prediction,
            "confidence": round(confidence, 4),
            "source_credibility": credibility
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# FILE INPUT
# -------------------------
@app.route("/predict-file", methods=["POST"])
def predict_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        text = file.read().decode("utf-8")

        if len(text) < 20:
            return jsonify({"error": "File content too small"}), 400

        result, confidence = analyze_text(text)

        return jsonify({
            "prediction": result,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

