"""
train.py
Train and evaluate the complaint category classifier.

    python train.py                          # uses data/complaints_clean.csv if present
    python train.py --data data/some.csv     # or any CSV with columns: category,text

Two models are compared on a held-out test set (20%) and the better one
(by macro F1) is saved to models/category_model.joblib together with
metrics.json and, if matplotlib is installed, a confusion matrix image.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config import CLEAN_DATA_PATH, METRICS_PATH, MODEL_PATH, SAMPLE_DATA_PATH
from textutils import clean_text


def build_pipelines(n_rows: int):
    min_df = 1 if n_rows < 1000 else 3
    tfidf = lambda: TfidfVectorizer(  # noqa: E731
        ngram_range=(1, 2), min_df=min_df, sublinear_tf=True,
        stop_words="english", max_features=100_000,
    )
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", tfidf()),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
    }


def train(data_path, model_path=MODEL_PATH, metrics_path=METRICS_PATH, seed=42):
    df = pd.read_csv(data_path).dropna(subset=["category", "text"])
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.2,
        stratify=df["category"], random_state=seed,
    )

    results, fitted = {}, {}
    for name, pipe in build_pipelines(len(df)).items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        }
        fitted[name] = pipe

    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    best = fitted[best_name]
    pred = best.predict(X_test)
    labels = sorted(df["category"].unique())

    metrics = {
        "data_file": str(data_path),
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": labels,
        "compared_models": results,
        "selected_model": best_name,
        "accuracy": results[best_name]["accuracy"],
        "macro_f1": results[best_name]["macro_f1"],
        "per_class": classification_report(
            y_test, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
    }

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best, "metrics_summary": {
        "selected_model": best_name, "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"], "n_total": metrics["n_total"]}},
        model_path)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2))
    _save_confusion_plot(metrics, Path(metrics_path).with_name("confusion_matrix.png"))
    return metrics


def _save_confusion_plot(metrics, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = metrics["classes"]
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix – {metrics['selected_model']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    data = Path(args.data) if args.data else (
        CLEAN_DATA_PATH if CLEAN_DATA_PATH.exists() else SAMPLE_DATA_PATH)
    if data == SAMPLE_DATA_PATH:
        print("NOTE: training on the tiny synthetic sample file. For real results run "
              "scripts/prepare_data.py on the CFPB dataset first.\n")

    m = train(data)
    print(f"Data: {m['data_file']}  ({m['n_total']:,} complaints, "
          f"{m['n_train']:,} train / {m['n_test']:,} test)")
    for name, r in m["compared_models"].items():
        print(f"  {name:20s} accuracy={r['accuracy']:.3f}  macro-F1={r['macro_f1']:.3f}")
    print(f"Selected: {m['selected_model']}")
    print(f"Saved model to {MODEL_PATH} and metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
