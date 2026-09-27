import json
import os

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from model import SentimentLSTM, PAD_TOKEN, UNK_TOKEN, CLASS_NAMES
from preprocessing import preprocess

BASE_DIR = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

EMOJI = {"Negative": "😞", "Neutral": "😐", "Positive": "😄"}

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

_model = None
_vocab = None
_config = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_artifacts():
    global _model, _vocab, _config
    with open(os.path.join(ARTIFACT_DIR, "config.json"), encoding="utf-8") as f:
        _config = json.load(f)
    with open(os.path.join(ARTIFACT_DIR, "vocab.json"), encoding="utf-8") as f:
        _vocab = json.load(f)

    model = SentimentLSTM(
        vocab_size=_config["vocab_size"],
        embed_dim=_config["embed_dim"],
        hidden_dim=_config["hidden_dim"],
        num_layers=_config["num_layers"],
        pad_idx=_vocab[PAD_TOKEN],
    )
    state = torch.load(os.path.join(ARTIFACT_DIR, "model.pt"), map_location=_device)
    model.load_state_dict(state)
    model.to(_device)
    model.eval()
    _model = model
    print("Model, vocab, and config loaded successfully.")


def numericalize(tokens, vocab, max_len):
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens][:max_len]
    length = max(len(ids), 1)
    ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids, length


def predict_sentiment(text: str):
    if _model is None:
        load_artifacts()

    tokens = preprocess(text)
    if not tokens:
        # Nothing meaningful left after cleaning (e.g. empty/punctuation-only input)
        return {
            "label": "Neutral",
            "emoji": EMOJI["Neutral"],
            "confidence": 0.0,
            "probabilities": {name: 0.0 for name in CLASS_NAMES},
            "tokens": [],
            "warning": "No meaningful words detected in the input.",
        }

    ids, length = numericalize(tokens, _vocab, _config["max_len"])
    ids_tensor = torch.tensor([ids], dtype=torch.long).to(_device)
    length_tensor = torch.tensor([length], dtype=torch.long).to(_device)

    with torch.no_grad():
        logits = _model(ids_tensor, length_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    label = CLASS_NAMES[pred_idx]

    return {
        "label": label,
        "emoji": EMOJI[label],
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))
        },
        "tokens": tokens,
    }


@app.route("/api/health", methods=["GET"])
def health():
    if _model is None:
        load_artifacts()
    return jsonify({
        "status": "ok",
        "vocab_size": _config["vocab_size"],
        "metrics": _config.get("metrics", {}),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please provide non-empty 'text'."}), 400
    if len(text) > 2000:
        return jsonify({"error": "Text is too long (max 2000 characters)."}), 400

    result = predict_sentiment(text)
    return jsonify(result)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


if __name__ == "__main__":
    load_artifacts()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
