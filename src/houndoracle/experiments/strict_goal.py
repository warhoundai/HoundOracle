"""Joint strict accuracy/precision goal; abstention cannot manufacture a passing score."""

from scipy.stats import beta

from .precision_gate import evaluate


def evaluate_goal(rows, *, target=0.99, minimum_coverage=0.35, alpha=0.05):
    """For one frozen policy on independent source/event components only.

    Bonferroni allocates alpha/2 to overall correctness and accepted-answer precision.
    This is not a universal guarantee, a training-score test or optional-stopping license.
    """
    selective = evaluate(
        rows, target_precision=target, minimum_coverage=minimum_coverage, alpha=alpha / 2
    )
    n = len(rows)
    correct = sum(not r.get("failure") and r["expected"] == r["decision"] for r in rows)
    lower = float(beta.ppf(alpha / 2, correct, n - correct + 1)) if correct else 0.0
    return {
        "count": n,
        "correct_all_decisions": correct,
        "accuracy_all_decisions": correct / n if n else None,
        "accuracy_lower_bound": lower,
        "target_accuracy": target,
        "family_alpha": alpha,
        "selective": selective,
        "gate_passed": bool(selective["gate_passed"] and lower >= target),
        "scope": "Joint fixed-policy held-out gate conditional on independent sampling/labels/span audit; does not certify arbitrary future questions or shifted retrieval distributions.",
    }
