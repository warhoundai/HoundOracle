"""Conservative finite-sample lower bound for paired binary outcome improvement."""

from scipy.stats import beta


def paired_improvement_lower(differences, *, alpha=0.05):
    """Bound P(candidate only correct) - P(baseline only correct).

    Each discordance marginal receives an exact binomial bound at alpha/2;
    the union bound gives at least1-alpha simultaneous coverage under iid pairs.
    Unlike a percentile bootstrap this does not degenerate to zero when no
    discordance was observed. It can be conservative and is not a shift guarantee.
    """
    values = list(differences)
    if not values or any(v not in (-1, 0, 1) for v in values) or not 0 < alpha < 1:
        raise ValueError("Require nonempty paired binary differences and valid alpha")
    n = len(values)
    positive, negative = values.count(1), values.count(-1)
    lower = float(beta.ppf(alpha / 2, positive, n - positive + 1)) if positive else 0.0
    upper = float(beta.ppf(1 - alpha / 2, negative + 1, n - negative)) if negative < n else 1.0
    return lower - upper
