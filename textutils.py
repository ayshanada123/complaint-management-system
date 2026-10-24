"""Text cleaning shared by data preparation, training and prediction."""

import re

_DATE = re.compile(r"\bX{1,2}/X{1,2}/(?:X{4}|\d{4})\b", re.IGNORECASE)  # XX/XX/2020
_REDACTED = re.compile(r"\bX{2,}\b", re.IGNORECASE)      # CFPB masks names/dates as XXXX
_AMOUNT = re.compile(r"\{\$[\d.,]+\}")                   # masked amounts like {$250.00}
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove CFPB redaction tokens and normalise whitespace."""
    if not isinstance(text, str):
        return ""
    text = _DATE.sub(" ", text)
    text = _REDACTED.sub(" ", text)
    text = _AMOUNT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
