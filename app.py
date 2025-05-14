import os
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# ─── Load model/tokenizer from Hugging Face ───
MODEL_REPO = "Promitsaha1/best_model_LLM_annotation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.eval()

# ─── Bias labels and per-label trigger thresholds ───
LABEL_COLS = [
    "Anchoring",
    "Illusory Truth Effect",
    "Information Overload",
    "Mere-Exposure Effect"
]
THRESHOLDS = {
    "Anchoring":              0.50,
    "Illusory Truth Effect":  0.80,
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

    # Predict logits and convert to probabilities
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze().tolist()

    # Determine which biases trigger based on per-label thresholds
    triggered = [
        LABEL_COLS[i]
        for i, p in enumerate(probs)
        if p >= THRESHOLDS[LABEL_COLS[i]]
    ]

    # Overall risk score: max probability among triggered cues, else 0
    if triggered:
        triggered_probs = [p for i, p in enumerate(probs) if LABEL_COLS[i] in triggered]
        risk_pct = max(triggered_probs) * 100
    else:
        risk_pct = 0.0

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
