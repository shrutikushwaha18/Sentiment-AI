# Sentiment-AI

Sentiment analysis using PyTorch, BiLSTM and Flask.

A working end-to-end sentiment analysis prototype: a PyTorch LSTM model
(Embedding → BiLSTM → Linear) trained to classify text as **Negative**,
**Neutral**, or **Positive**, served through a Flask API, with a
standalone frontend that shows the result as a label, a confidence
percentage, and an emoji.

A trained model is already included in `backend/artifacts/`, so the app
runs out of the box — no training required to try it.

```
sentimentai/
├── backend/
│   ├── app.py              Flask API (serves predictions + the frontend)
│   ├── model.py             Model architecture (Embedding + BiLSTM + Linear)
│   ├── preprocessing.py     Cleaning / tokenizing / stemming pipeline
│   ├── train.py             Training script (Dataset/DataLoader/Adam/CrossEntropyLoss)
│   ├── requirements.txt
│   ├── data/
│   │   ├── generate_dataset.py   Builds the sample training dataset
│   │   └── dataset.csv           720 labeled example sentences (already generated)
│   └── artifacts/           Trained model + vocab + metrics (already generated)
│       ├── model.pt
│       ├── vocab.json
│       └── config.json
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## Quick start (run it locally)

You need Python 3.9+.

```bash
cd sentimentai/backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000** — the Flask server serves both the API and
the frontend, so that's the only URL you need.

> First-time note: `torch` is a fairly large dependency (a few hundred MB
> to a couple GB depending on your platform). This is normal and only
> happens once.

## Retraining the model (optional)

The repo already ships a trained model, but you can regenerate everything:

```bash
cd backend
python data/generate_dataset.py   # rebuilds data/dataset.csv (optional — it's already there)
python train.py                    # trains and overwrites artifacts/
```

`train.py` prints accuracy, precision, recall, F1-score, and a confusion
matrix at the end of training, and saves them into `artifacts/config.json`
(the frontend reads these to show live model stats).

To use your own data instead of the bundled synthetic dataset, replace
`backend/data/dataset.csv` with a CSV that has `text,label` columns, where
`label` is `0` (negative), `1` (neutral), or `2` (positive).

## API

**POST `/api/predict`**
```json
{ "text": "I absolutely loved this, it exceeded every expectation!" }
```
Response:
```json
{
  "label": "Positive",
  "emoji": "😄",
  "confidence": 98.93,
  "probabilities": { "Negative": 1.04, "Neutral": 0.03, "Positive": 98.93 },
  "tokens": ["absolute", "lov", "exceed", "everi", "expect"]
}
```

**GET `/api/health`** — returns model status and saved evaluation metrics.

## Deploying it

**Simplest option — one host serves both.** Flask already serves the
frontend, so any Python host works as a single deployment:
- Render / Railway / Fly.io: point them at `backend/`, install
  `requirements.txt`, and run `gunicorn app:app` (add `gunicorn` to
  requirements.txt for production) or `python app.py`.
- Set the `PORT` environment variable if your host requires a specific one
  (the app already reads `PORT` from the environment).

**Split option — separate static frontend + API.** Deploy `frontend/` to
any static host (Netlify, Vercel, GitHub Pages, S3) and `backend/` to a
Python host. Then open `frontend/script.js` and set:
```js
const API_BASE = "https://your-backend-url.com";
```
and make sure `flask-cors` (already included) stays enabled so the two
origins can talk to each other.

## Notes on the model

This ships with a synthetic-but-realistic dataset (`data/dataset.csv`,
720 examples, template-generated + hand-curated) so the prototype works
immediately without needing an external dataset. It reaches ~96% validation
accuracy on its own held-out split, but for a production system you'd want
to swap in a real labeled dataset (e.g. IMDB reviews, SST, or your own
labeled text) — just drop it into `data/dataset.csv` with the same
`text,label` format and rerun `train.py`.
