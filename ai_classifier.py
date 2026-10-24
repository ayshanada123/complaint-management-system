"""
ai_classifier.py
Two pieces:

1. classify_category(text) -> (category, confidence)
   A TF-IDF + Logistic Regression / Naive Bayes model trained by train.py on
   the CFPB consumer complaint data. If no trained model file exists, a small
   fallback model is trained on data/sample_complaints.csv so the app still
   runs (MODEL_SOURCE tells you which one is in use).

2. analyze_priority(text) -> (priority, sentiment)
   A transparent rule-based engine: negative-word and urgency-phrase counts.
"""

import json
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import (LOW_CONFIDENCE_THRESHOLD, METRICS_PATH, MODEL_PATH,
                    SAMPLE_DATA_PATH)
from textutils import clean_text

_model = None
MODEL_SOURCE = None  # "trained model" or "sample-data fallback"


def _load_model():
    global _model, MODEL_SOURCE
    if _model is not None:
        return _model
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)["pipeline"]
        MODEL_SOURCE = "trained model"
    else:
        df = pd.read_csv(SAMPLE_DATA_PATH)
        _model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]).fit(df["text"].map(clean_text), df["category"])
        MODEL_SOURCE = "sample-data fallback"
    return _model


def reload_model():
    """Forget the cached model (used after re-training and in tests)."""
    global _model, MODEL_SOURCE
    _model, MODEL_SOURCE = None, None


def model_info():
    """Details about the model in use, for the console and /api/model-info."""
    _load_model()
    info = {"source": MODEL_SOURCE}
    if METRICS_PATH.exists() and MODEL_SOURCE == "trained model":
        m = json.loads(METRICS_PATH.read_text())
        info.update({
            "selected_model": m["selected_model"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "n_train": m["n_train"],
            "n_test": m["n_test"],
        })
    return info


def classify_category(text: str):
    """Return (category, confidence) for a complaint description."""
    model = _load_model()
    proba = model.predict_proba([clean_text(text)])[0]
    best = proba.argmax()
    return str(model.classes_[best]), round(float(proba[best]), 3)


# ---------------------------------------------------------------------------
# Sentiment + urgency -> priority
# ---------------------------------------------------------------------------

_NEGATIVE_WORDS = {
    "angry", "furious", "terrible", "horrible", "awful", "worst", "unacceptable",
    "disgusted", "disappointed", "frustrated", "useless", "broken", "scam",
    "rude", "ridiculous", "waste", "never", "refuse", "refused", "ignored",
    "poor", "unprofessional", "fail", "failed", "failing", "problem", "issue",
    "wrong", "damaged", "delayed", "missing", "hate", "nobody", "waiting",
    "overcharged", "double", "unresolved", "complain", "complaining", "twice",
    "fraud", "fraudulent", "unauthorized", "stolen", "harassing", "harassment",
    "threatened", "denied", "incorrect", "mistake",
}

_POSITIVE_WORDS = {
    "thanks", "thank", "great", "good", "appreciate", "love", "excellent",
    "happy", "pleased", "helpful", "quick", "resolved", "smooth",
}

_URGENCY_PHRASES = [
    "urgent", "urgently", "immediately", "asap", "now", "emergency", "legal",
    "lawyer", "attorney", "lawsuit", "sue", "today", "critical", "foreclosure",
    "eviction", "garnish", "identity theft", "third time", "still not",
    "no response", "cancel my account", "refund now",
]
# Whole-word / whole-phrase match, so "now" does not fire inside "know".
_URGENCY_REGEX = [re.compile(r"\b" + re.escape(p) + r"\b") for p in _URGENCY_PHRASES]


def _tokenize(text: str):
    return re.findall(r"[a-z']+", text.lower())


def analyze_priority(text: str):
    """Return (priority, sentiment) for a complaint description."""
    tokens = set(_tokenize(text))
    lowered = text.lower()

    neg_hits = len(tokens & _NEGATIVE_WORDS)
    pos_hits = len(tokens & _POSITIVE_WORDS)
    urgency_hits = sum(1 for rx in _URGENCY_REGEX if rx.search(lowered))

    if neg_hits > pos_hits:
        sentiment = "Negative"
    elif pos_hits > neg_hits:
        sentiment = "Positive"
    else:
        sentiment = "Neutral"

    # Caps stop very long complaints from always scoring "Urgent".
    score = min(urgency_hits, 2) * 2 + min(neg_hits, 3) - min(pos_hits, 2)

    if score >= 4:
        priority = "Urgent"
    elif score >= 2:
        priority = "High"
    elif score >= 1:
        priority = "Medium"
    else:
        priority = "Low"
    return priority, sentiment


def classify_complaint(text: str):
    """Everything the app needs for one complaint."""
    category, confidence = classify_category(text)
    priority, sentiment = analyze_priority(text)
    return {
        "category": category,
        "category_confidence": confidence,
        "needs_review": confidence < LOW_CONFIDENCE_THRESHOLD,
        "priority": priority,
        "sentiment": sentiment,
    }
