"""Explicit logical negation views for declared propositions; no truth calibration."""

import numpy as np


def negated_view(proposition, *, declared_proposition=False):
    if declared_proposition is not True:
        raise ValueError(
            "Require an explicitly declared proposition; do not negate an unparsed question"
        )
    if not isinstance(proposition, str) or not proposition.strip():
        raise ValueError("Nonempty proposition required")
    # The complete original text remains inside the scope of NOT.
    return "It is not true that " + proposition


def combine(original_logits, negated_logits):
    """Class order NO,YES,MAYBE. NOT swaps entailment/refutation, preserves unknown.

    Related model views are correlated, not independent evidence. The resulting
    vector is an uncalibrated relation score, never a probability of factual truth.
    """

    def softmax(values):
        z = np.asarray(values, dtype=float)
        if z.shape != (3,) or not np.isfinite(z).all():
            raise ValueError("Three finite relation logits required")
        p = np.exp(z - z.max())
        return p / p.sum()

    a, b = softmax(original_logits), softmax(negated_logits)[[1, 0, 2]]
    p = (a + b) / 2
    return {
        "candidate": ["NO", "YES", "MAYBE"][int(p.argmax())],
        "raw_score": float(p.max()),
        "relation_scores": p.tolist(),
        "views_agree": int(a.argmax()) == int(b.argmax()),
        "calibrated_correctness": None,
        "production_qualified": False,
    }
