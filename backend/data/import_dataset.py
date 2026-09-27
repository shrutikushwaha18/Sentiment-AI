"""
import_dataset.py
------------------
Converts a Kaggle-style sentiment CSV into the text,label format train.py
expects (label: 0=negative, 1=neutral, 2=positive).

Handles common column names and label encodings automatically:
  - text column:  text, review, Text, sentence, tweet, Sentence, content, Review
  - label column: label, sentiment, target, Sentiment, Label
  - label values:
      strings   -> "negative"/"neg"/"0" style, "neutral"/"neu", "positive"/"pos"/"1"
      Sentiment140 style -> 0=negative, 2=neutral, 4=positive
      -1/0/1 style -> -1=negative, 0=neutral, 1=positive
      0/1 binary (no neutral) -> 0=negative, 1=positive

Usage:
    python import_dataset.py path/to/kaggle_sentiment.csv
    python import_dataset.py path/to/kaggle_sentiment.csv --text-col review --label-col Sentiment

The original synthetic dataset is backed up to dataset_synthetic_backup.csv
the first time you run this, so you can always go back or blend the two.
"""

import argparse
import csv
import os
import shutil
import sys

HERE = os.path.dirname(__file__)
OUT_PATH = os.path.join(HERE, "dataset.csv")
BACKUP_PATH = os.path.join(HERE, "dataset_synthetic_backup.csv")

TEXT_COL_CANDIDATES = [
    "text", "review", "sentence", "tweet", "content", "comment",
    "Text", "Review", "Sentence", "Tweet", "Content", "Comment",
    "review_text", "OriginalTweet",
]
LABEL_COL_CANDIDATES = [
    "label", "sentiment", "target", "class",
    "Label", "Sentiment", "Target", "Class",
]

NEG_STRINGS = {"negative", "neg", "0", "bad"}
NEU_STRINGS = {"neutral", "neu"}
POS_STRINGS = {"positive", "pos", "4", "good"}


def detect_column(fieldnames, candidates):
    field_lower = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c in fieldnames:
            return c
        if c.lower() in field_lower:
            return field_lower[c.lower()]
    return None


def normalize_label(raw, seen_values):
    """Map a raw label value to 0 (neg), 1 (neu), or 2 (pos), or None if unrecognized."""
    val = str(raw).strip()
    seen_values.add(val)
    low = val.lower()

    if low in NEG_STRINGS:
        return 0
    if low in NEU_STRINGS:
        return 1
    if low in POS_STRINGS:
        return 2

    # Numeric encodings
    try:
        num = float(val)
    except ValueError:
        return None

    if num in (-1,):
        return 0
    if num in (0,):
        return 0
    if num in (1,):
        return 2  # binary datasets: 0=neg, 1=pos (no neutral class)
    if num in (2,):
        return 1  # e.g. Sentiment140 neutral=2, or 1-5 star midpoint
    if num in (3,):
        return 1
    if num in (4,):
        return 2
    if num in (5,):
        return 2
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to the Kaggle sentiment CSV")
    parser.add_argument("--text-col", default=None, help="Override text column name")
    parser.add_argument("--label-col", default=None, help="Override label column name")
    parser.add_argument("--max-rows", type=int, default=20000,
                         help="Cap rows for faster training on CPU (default 20000)")
    parser.add_argument("--encoding", default="utf-8",
                         help="File encoding, try 'latin-1' if you get decode errors")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"File not found: {args.input_csv}")
        sys.exit(1)

    if os.path.exists(OUT_PATH) and not os.path.exists(BACKUP_PATH):
        shutil.copy(OUT_PATH, BACKUP_PATH)
        print(f"Backed up existing dataset.csv -> {BACKUP_PATH}")

    with open(args.input_csv, newline="", encoding=args.encoding, errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        text_col = args.text_col or detect_column(fieldnames, TEXT_COL_CANDIDATES)
        label_col = args.label_col or detect_column(fieldnames, LABEL_COL_CANDIDATES)

        if not text_col or not label_col:
            print(f"Could not auto-detect columns. Found columns: {fieldnames}")
            print("Re-run with --text-col and --label-col to specify them manually, e.g.:")
            print(f'  python import_dataset.py "{args.input_csv}" --text-col review --label-col sentiment')
            sys.exit(1)

        print(f"Using text column: '{text_col}', label column: '{label_col}'")

        rows = []
        skipped = 0
        seen_values = set()
        for row in reader:
            text = (row.get(text_col) or "").strip()
            raw_label = row.get(label_col)
            if not text or raw_label is None or str(raw_label).strip() == "":
                skipped += 1
                continue
            label = normalize_label(raw_label, seen_values)
            if label is None:
                skipped += 1
                continue
            rows.append((text, label))
            if len(rows) >= args.max_rows:
                break

    if not rows:
        print("No usable rows found. Raw label values seen:", seen_values)
        sys.exit(1)

    counts = {0: 0, 1: 0, 2: 0}
    for _, label in rows:
        counts[label] += 1

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT_PATH} (skipped {skipped} unusable rows)")
    print(f"Negative: {counts[0]}, Neutral: {counts[1]}, Positive: {counts[2]}")
    if counts[1] == 0:
        print("\nNote: no neutral examples were found in this dataset — it looks binary "
              "(positive/negative only). The model will still train, but it won't have "
              "learned a real 'neutral' class. Consider blending in some neutral examples "
              "from dataset_synthetic_backup.csv if you want that class to work well.")
    print("\nNext step: run `python train.py` to retrain on this data.")


if __name__ == "__main__":
    main()
