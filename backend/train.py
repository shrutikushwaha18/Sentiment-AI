"""
train.py
--------
Full training pipeline:
  raw CSV -> preprocessing -> vocabulary -> numericalize -> pad
  -> PyTorch Dataset/DataLoader -> Embedding+LSTM+Linear model
  -> CrossEntropyLoss + Adam -> train -> evaluate -> save artifacts

Run:  python train.py
Produces (in ./artifacts/):
  model.pt        - trained model weights
  vocab.json       - word -> id mapping
  config.json       - model hyperparameters + metrics
"""

import csv
import json
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report,
)

from model import SentimentLSTM, PAD_TOKEN, UNK_TOKEN, CLASS_NAMES
from preprocessing import preprocess

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MAX_LEN = 30
MIN_FREQ = 1
BATCH_SIZE = 16
EPOCHS = 18
LR = 1e-3
EMBED_DIM = 100
HIDDEN_DIM = 128


def load_data(path):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def build_vocab(token_lists, min_freq=1):
    freq = {}
    for tokens in token_lists:
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def numericalize(tokens, vocab, max_len):
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens][:max_len]
    length = max(len(ids), 1)
    ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids, length


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.samples = []
        for text, label in zip(texts, labels):
            tokens = preprocess(text)
            ids, length = numericalize(tokens, vocab, max_len)
            self.samples.append((torch.tensor(ids, dtype=torch.long),
                                  torch.tensor(length, dtype=torch.long),
                                  torch.tensor(label, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ids, lengths, labels in loader:
            ids, lengths = ids.to(device), lengths.to(device)
            logits = model(ids, lengths)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist()
    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    texts, labels = load_data(DATA_PATH)
    print(f"Loaded {len(texts)} examples")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    train_tokens = [preprocess(t) for t in train_texts]
    vocab = build_vocab(train_tokens, min_freq=MIN_FREQ)
    print(f"Vocabulary size: {len(vocab)}")

    train_ds = SentimentDataset(train_texts, train_labels, vocab, MAX_LEN)
    test_ds = SentimentDataset(test_texts, test_labels, vocab, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SentimentLSTM(
        vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=1, pad_idx=vocab[PAD_TOKEN],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1 = -1
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for ids, lengths, batch_labels in train_loader:
            ids, lengths, batch_labels = ids.to(device), lengths.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(ids, lengths)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * ids.size(0)

        avg_loss = total_loss / len(train_ds)
        metrics = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:2d}/{EPOCHS} | loss {avg_loss:.4f} | "
              f"val_acc {metrics['accuracy']:.3f} | val_f1 {metrics['f1']:.3f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = metrics

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, test_loader, device)
    print("\n=== Final evaluation (best checkpoint) ===")
    print(final_metrics["report"])
    print("Confusion matrix (rows=true, cols=pred) [Neg, Neu, Pos]:")
    for row in final_metrics["confusion_matrix"]:
        print(row)

    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "model.pt"))
    with open(os.path.join(ARTIFACT_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    config = {
        "vocab_size": len(vocab),
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": 1,
        "max_len": MAX_LEN,
        "class_names": CLASS_NAMES,
        "metrics": {
            "accuracy": final_metrics["accuracy"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1": final_metrics["f1"],
            "confusion_matrix": final_metrics["confusion_matrix"],
        },
    }
    with open(os.path.join(ARTIFACT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved model, vocab, and config to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
