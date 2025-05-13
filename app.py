import os
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# ─── Load from your Hugging Face repo ───
MODEL_REPO = "Promitsaha1/best_model_LLM_annotation"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_REPO)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)

# ─── Your four bias labels ───
LABEL_COLS = [
    "Anchoring",
    "Illusory Truth Effect",
    "Information Overload",
    "Mere-Exposure Effect"
]

def compute_phishing_risk(body: str):
    # Tokenize & run model
    inputs = tokenizer(
        body,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.sigmoid(logits).squeeze().tolist()  # [p_anchor, p_illusory, p_info, p_mere]

    # Risk score: highest probability ×100
    risk_pct = max(probs) * 100

    # Which biases triggered at ≥50%?
    triggered = [LABEL_COLS[i] for i, p in enumerate(probs) if p >= 0.5]

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
    # Railway (and many hosts) set the PORT env var for you
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
