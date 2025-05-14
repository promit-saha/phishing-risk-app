import os
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib

app = Flask(__name__)

# ─── Load model/tokenizer from Hugging Face ───
MODEL_REPO = "Promitsaha1/best_model_LLM_annotation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.eval()

# ─── Load pre-fitted calibrator (created offline) ───
CALIBRATOR_PATH = "calibrator.joblib"
calibrator = joblib.load(CALIBRATOR_PATH)

# ─── Bias labels and per-label trigger thresholds ───
LABEL_COLS = [
    "Anchoring",
    "Illusory Truth Effect",
    "Information Overload",
    "Mere-Exposure Effect"
]
THRESHOLDS = {
    "Anchoring":              0.50,
    "Illusory Truth Effect":  0.70,
    "Information Overload":   0.50,
    "Mere-Exposure Effect":   0.50
}

def compute_phishing_risk(body: str):
    # Tokenize input text
    inputs = tokenizer(
        body,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    # Raw model logits
    with torch.no_grad():
        logits = model(**inputs).logits

    # Calibrated probabilities (expects shape [1, num_labels])
    probs = calibrator.predict_proba(logits.cpu().numpy())[0].tolist()

    # Overall risk score: highest bias probability ×100
    risk_pct = max(probs) * 100

    # Only trigger the top-scoring bias if above its threshold
    max_idx = int(torch.argmax(logits))
    triggered = []
    if probs[max_idx] >= THRESHOLDS[LABEL_COLS[max_idx]]:
        triggered = [LABEL_COLS[max_idx]]

    return risk_pct, dict(zip(LABEL_COLS, probs)), triggered

@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "body":      "",
        "risk":      None,
        "probs":     None,
        "triggered": None
    }

    if request.method == "POST":
        body = request.form["body"]
        risk, probs, triggered = compute_phishing_risk(body)
        context.update({
            "body":      body,
            "risk":      f"{risk:.1f}",
            "probs":     {k: f"{v:.3f}" for k, v in probs.items()},
            "triggered": triggered
        })

    return render_template("index.html", **context)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
