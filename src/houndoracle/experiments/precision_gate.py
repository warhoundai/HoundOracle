"""Declared high-precision acceptance gate; a confidence score is not a truth guarantee."""

import math

from scipy.stats import beta


def evaluate(rows, *, target_precision=0.99, minimum_coverage=0.30, alpha=0.05):
    if not 0 < target_precision < 1 or not 0 < minimum_coverage <= 1 or not 0 < alpha < 1:
        raise ValueError("Valid nontrivial precision, coverage and alpha required")
    if len({r["id"] for r in rows}) != len(rows) or len({r["group_id"] for r in rows}) != len(rows):
        raise ValueError("Require one independently sampled query per source/event component")
    if any(
        r["expected"] not in ["YES", "NO", "MAYBE", "INVALID"]
        or r["decision"] not in ["YES", "NO", "MAYBE", "INVALID"]
        for r in rows
    ):
        raise ValueError("Unknown verdict")
    valid = [r for r in rows if r["expected"] != "INVALID"]
    accepted = [r for r in valid if r["decision"] in ["YES", "NO"]]
    errors = sum(bool(r.get("failure")) or r["decision"] != r["expected"] for r in accepted)
    n = len(accepted)
    upper = float(beta.ppf(1 - alpha, errors + 1, n - errors)) if n and errors < n else 1.0
    confidence_contract = all(
        isinstance(r.get("confidence"), (int, float))
        and not isinstance(r.get("confidence"), bool)
        and math.isfinite(r["confidence"])
        and target_precision <= r["confidence"] <= 1
        for r in accepted
    )
    audited = bool(accepted) and all(
        r.get("citation_supported") is True
        and r.get("citation_audit_method") == "independent_span_review"
        for r in accepted
    )
    coverage = n / len(valid) if valid else 0
    passed = bool(
        n
        and upper <= 1 - target_precision
        and coverage >= minimum_coverage
        and audited
        and confidence_contract
    )
    return {
        "target_precision": target_precision,
        "minimum_coverage": minimum_coverage,
        "alpha": alpha,
        "valid_questions": len(valid),
        "accepted": n,
        "errors": errors,
        "coverage": coverage,
        "correct_definitive_fraction": (n - errors) / len(valid) if valid else 0,
        "empirical_precision": 1 - errors / n if n else None,
        "one_sided_error_upper": upper,
        "citations_independently_audited": audited,
        "confidence_contract_satisfied": confidence_contract,
        "gate_passed": passed,
        "scope": "Only under independently sampled/exchangeable source-event components and independent ground truth/span audit. Fixed selection policy; no tuning on this evaluation or optional-stopping guarantee. Not an individual-answer truth probability or distribution-shift guarantee.",
    }
