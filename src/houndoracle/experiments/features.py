"""Cheap contextual features for a research head; none are symbolic truth rules."""

import math
import re

SCHEMA = "lexical-context-v1"


def context_features(question, evidence):
    passage = "\n".join(e["text"] for e in evidence)
    words = lambda text: set(re.findall(r"\b\w+\b", text.casefold()))
    claim_words, evidence_words = words(question), words(passage)
    numbers = lambda text: set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    qnums, enums = numbers(question), numbers(passage)
    negative = {"not", "never", "no", "neither", "without"}
    return [
        len(claim_words & evidence_words) / max(1, len(claim_words)),
        len(claim_words & evidence_words) / max(1, len(claim_words | evidence_words)),
        math.log1p(len(claim_words)),
        math.log1p(len(evidence_words)),
        math.log1p(len(qnums)),
        math.log1p(len(enums)),
        len(qnums & enums) / max(1, len(qnums)),
        float(bool(claim_words & negative)),
        float(bool(evidence_words & negative)),
        float('"' in question),
        math.log1p(len(evidence)),
    ]
