"""Experimental document context; source titles are hints, never verified facts."""

import re
import unicodedata
from urllib.parse import unquote

SCHEMA = "source-context-v2"


def document_title(item):
    value = item.get("page")
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFC", unquote(value)).replace("_", " ").strip()[:300]


class SourceContextNLI:
    def __init__(self, model):
        self.model = model
        self.config = model.config | {"evidence_serialization": SCHEMA}

    def predict(self, question, evidence):
        contextual = []
        for item in evidence:
            title = document_title(item)
            contextual.append(
                item
                | {
                    "text": f"Document title: {title}\nPassage: {item['text']}"
                    if title and item["text"].strip()
                    else item["text"]
                }
            )
        return self.model.predict(question, contextual)


def grounding_features(question, evidence):
    def tokens(value):
        return set(re.findall(r"\w+", value.casefold()))

    q = tokens(question)
    titles = [tokens(document_title(item)) for item in evidence]
    overlap = [len(q & t) / max(1, len(t)) for t in titles]
    return [
        max(overlap, default=0.0),
        sum(overlap) / max(1, len(overlap)),
        sum(bool(t) for t in titles) / max(1, len(titles)),
        float(any(t and t <= q for t in titles)),
    ]
