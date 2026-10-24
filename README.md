# AI-Assisted Complaint Management System

A full-stack Flask web app that receives customer complaints, **classifies them
with a machine-learning model trained on real complaints**, scores priority,
routes each complaint to an agent with an SLA deadline, and lets support staff
manage the queue from a login-protected console.

## Features

- **Customer portal** – submit a complaint, get a ticket number, track its status.
  Live AI preview of the predicted category and priority while typing.
- **ML category classifier** – TF-IDF + Logistic Regression (compared against Naive
  Bayes; the better model by macro-F1 is kept), trained on the public
  [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).
- **Priority engine** – transparent rule-based scoring from negative-sentiment words and
  urgency phrases (whole-word matching, capped so long texts don't always become "Urgent").
- **Low-confidence flag** – predictions below a confidence threshold are marked for human review.
- **Routing + SLA** – category → agent; priority → deadline (Urgent 4h … Low 120h).
  Changing priority recalculates the deadline.
- **Staff console** – login required; filter by status/category/priority, overdue
  highlighting, update status/priority, resolution notes.
- **JSON API** – `/api/complaints`, `/api/stats` (staff only), `/api/classify-preview`, `/api/model-info`.
- **Security basics** – session login with hashed password, CSRF tokens on every POST,
  parameterised SQL, input validation, safe redirects.
- **Tests + CI** – 27 unit/integration tests, run on every push via GitHub Actions.

## Architecture

```
Browser ──> Flask (app.py) ──> database.py ──> SQLite (crm.db)
              │
              └──> ai_classifier.py ──> models/category_model.joblib   (made by train.py)
                                          ▲
data/raw/complaints.csv ─> scripts/prepare_data.py ─> data/complaints_clean.csv ─> train.py
```

## Dataset

The category model is trained on the **CFPB Consumer Complaint Database**
(US government, public domain). The copy used here was downloaded from Kaggle:
"Consumer Complaint Database" by Kaushik D (CC0: Public Domain), which republishes
the CFPB data. Only complaints that include a consumer narrative are used. The CFPB
`Product` field is mapped to 7 categories: Credit Reporting, Debt Collection, Mortgage,
Credit & Prepaid Cards, Bank Accounts & Services, Loans, Money Transfers. Redaction
tokens (`XXXX`, masked dates and amounts) are stripped. From the full file, 4,000
complaints per category were sampled (28,000 in total), with 20% held out for testing.s

`data/sample_complaints.csv` is a tiny **synthetic** file used only so the app and tests run
without downloading anything. It is not used for the reported results.

## Quick start

```bash
git clone <your-repo-url> && cd crm-mini
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Download complaints.csv from the CFPB site (or Kaggle) into data/raw/
# 2. Build a clean, balanced training file
python scripts/prepare_data.py data/raw/complaints.csv --per-class 4000
# 3. Train and evaluate (prints accuracy / macro-F1, saves models/metrics.json)
python train.py
# 4. Run the app
cp .env.example .env     # then edit the values
python app.py            # http://127.0.0.1:5000
```

Without steps 1–3 the app still runs using the small synthetic fallback model
(the console sidebar shows which model is active).

Customer portal: `/submit` · Staff console: `/console` (log in first).

## Model results

Run `python train.py`; results are written to `models/metrics.json` and
`models/confusion_matrix.png`.

| Metric | Value |
|---|---|
| Accuracy (20% held-out test set) | 82.2% |
| Macro-F1 | 0.822 |

## Configuration

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Signs session cookies |
| `STAFF_USERNAME`, `STAFF_PASSWORD` | Console login (dev defaults are used if unset) |
| `CRM_DB_PATH` | Location of the SQLite file |

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

## Project structure

```
app.py                 Flask routes, login, CSRF
config.py              Settings, SLA table, agent routing
database.py            SQLite data access (parameterised queries)
ai_classifier.py       Category model loading + priority rules
train.py               Train / compare / evaluate / save model
scripts/prepare_data.py  CFPB CSV -> clean training CSV
textutils.py           Shared text cleaning
templates/ static/     HTML + CSS
tests/                 Unit and integration tests
```

## Limitations and next steps

- Priority is rule-based, not learned; a labelled urgency dataset would allow a trained model.
- SQLite suits a single-server prototype; use MySQL/PostgreSQL for many concurrent users.
- One shared staff account; add per-user accounts and roles.
- No email notifications yet.
- Not deployed; a Dockerfile and a production server (gunicorn) are the next step.
- Categories come from financial-services complaints, so the model targets that domain.
