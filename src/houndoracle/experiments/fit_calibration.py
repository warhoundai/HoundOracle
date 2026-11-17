"""Fit on disjoint components; validate mappings; certify selection on untouched groups."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import beta


def sigmoid_feature(score):
    return float(logit(min(1 - 1e-7, max(1e-7, score))))


def fit(method, rows):
    rows = [r for r in rows if r["candidate"] in ("YES", "NO") and r["raw_score"] is not None]
    if len(rows) < 10:
        raise ValueError("Too few eligible calibration examples")
    x = np.array([r["raw_score"] for r in rows])
    y = np.array([r["candidate"] == r["expected"] for r in rows], dtype=float)
    if method == "temperature":
        # Binary correctness temperature, not softmax temperature: does not change verdict.
        logits = logit(np.clip(x, 1e-7, 1 - 1e-7))

        def loss(log_t):
            p = expit(logits / np.exp(log_t))
            return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))

        result = minimize_scalar(loss, bounds=(-4, 4), method="bounded")
        return {"method": method, "temperature": float(np.exp(result.x)), "loss": float(result.fun)}
    if method == "logistic":
        from sklearn.linear_model import LogisticRegression

        if len(set(y)) < 2:
            raise ValueError("Logistic calibration needs both correctness classes")
        model = LogisticRegression(C=1, max_iter=500, random_state=1729).fit(
            logit(np.clip(x, 1e-7, 1 - 1e-7)).reshape(-1, 1), y
        )
        return {
            "method": method,
            "weight": float(model.coef_[0, 0]),
            "bias": float(model.intercept_[0]),
        }
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip").fit(x, y)
        return {
            "method": method,
            "x": model.X_thresholds_.tolist(),
            "y": model.y_thresholds_.tolist(),
        }
    raise ValueError("Unknown calibration method")


def predict(artifact, score):
    if score is None:
        return None
    if artifact["method"] == "temperature":
        return float(expit(sigmoid_feature(score) / artifact["temperature"]))
    if artifact["method"] == "logistic":
        return float(expit(artifact["weight"] * sigmoid_feature(score) + artifact["bias"]))
    return float(np.interp(score, artifact["x"], artifact["y"]))


def diagnostics(rows, mapping=None):
    eligible = [r for r in rows if r["candidate"] in ("YES", "NO") and r["raw_score"] is not None]
    if not eligible:
        return {"eligible": 0, "brier": None}
    p = np.array(
        [predict(mapping, r["raw_score"]) if mapping else r["raw_score"] for r in eligible]
    )
    correct = np.array([r["candidate"] == r["expected"] for r in eligible], dtype=float)
    bins = []
    ece = 0
    for lower in np.arange(0, 1, 0.1):
        mask = (p >= lower) & (p < (lower + 0.1) if lower < 0.9 else p <= 1)
        if mask.any():
            confidence = float(p[mask].mean())
            accuracy = float(correct[mask].mean())
            bins.append(
                {
                    "lower": float(lower),
                    "count": int(mask.sum()),
                    "mean_confidence": confidence,
                    "empirical_correctness": accuracy,
                }
            )
            ece += float(mask.mean()) * abs(confidence - accuracy)
    curve = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
        mask = p >= threshold
        curve.append(
            {
                "threshold": threshold,
                "coverage": int(mask.sum()) / len(rows),
                "selective_risk": float(1 - correct[mask].mean()) if mask.any() else None,
            }
        )
    return {
        "eligible": len(eligible),
        "brier": float(np.mean((p - correct) ** 2)),
        "ece_10_fixed_bins": ece,
        "reliability_bins": bins,
        "risk_coverage": curve,
        "note": "Correctness calibration among candidate definitive answers; not a per-answer guarantee.",
    }


def certify(mapping, risk_rows, *, threshold=0.8, alpha=0.05):
    selected = [
        r
        for r in risk_rows
        if r["candidate"] in ("YES", "NO") and predict(mapping, r["raw_score"]) >= threshold
    ]
    groups = {}
    for row in selected:
        groups.setdefault(row["group_id"], []).append(row)
    # Conservative group event: any incorrect accepted claim makes the entire group fail.
    successes = sum(
        all(r["candidate"] == r["expected"] for r in group) for group in groups.values()
    )
    n = len(groups)
    lower = float(beta.ppf(alpha, successes, n - successes + 1)) if successes else 0.0
    return {
        "threshold": threshold,
        "alpha": alpha,
        "selected_rows": len(selected),
        "selected_groups": n,
        "fully_correct_groups": successes,
        "group_correctness_lower_bound": lower,
        "qualified": lower >= threshold,
        "method": "one-sided Clopper-Pearson on group-any-error events",
        "assumptions": "Independent/exchangeable sampled components; does not guarantee row-weighted risk for arbitrary group sizes or distribution shift.",
        "row_weighted_risk_guaranteed": False,
    }
