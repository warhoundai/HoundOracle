"""Outcome, coverage, grouped uncertainty and paired comparisons; unknown costs stay unknown."""

import math
import random
import statistics


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lo = int(index)
    hi = min(len(values) - 1, lo + 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summary(rows):
    if not rows:
        return {"count": 0}
    valid = [r for r in rows if r["expected"] != "INVALID"]
    definitive = [r for r in valid if r["decision"] in ("YES", "NO") and not r.get("failure")]
    correct = sum(r["decision"] == r["expected"] for r in definitive)
    known_costs = [r.get("all_in_cost_usd") for r in rows]
    total_cost = sum(known_costs) if all(v is not None for v in known_costs) else None
    latencies = [r.get("latency_seconds") for r in rows]
    observed = [
        v
        for v in latencies
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
    ]
    complete_latency = len(observed) == len(rows)
    return {
        "count": len(rows),
        "groups": len({r["group_id"] for r in rows}),
        "accuracy_all_decisions": sum(
            not r.get("failure") and r["decision"] == r["expected"] for r in rows
        )
        / len(rows),
        "definitive_coverage": len(definitive) / len(valid) if valid else None,
        "correct_definitive_fraction": correct / len(valid) if valid else None,
        "selective_risk": 1 - correct / len(definitive) if definitive else None,
        "definitive_count": len(definitive),
        "correct_definitive_count": correct,
        "citation_supported_definitive_fraction": sum(
            r.get("citation_supported") is True for r in definitive
        )
        / len(definitive)
        if definitive
        and all(
            r.get("citation_supported") is not None
            and r.get("citation_audit_method") == "independent_span_review"
            for r in definitive
        )
        else None,
        "all_in_cost_usd": total_cost,
        "cost_per_correct_definitive_usd": total_cost / correct
        if total_cost is not None and correct
        else None,
        "latency_seconds": {
            "p50": statistics.median(observed) if complete_latency else None,
            "p95": percentile(observed, 0.95) if complete_latency else None,
            "p99": percentile(observed, 0.99) if complete_latency else None,
        },
        "observed_latency_count": len(observed),
        "missing_latency_count": len(rows) - len(observed),
        "cache_hits": sum(bool(r.get("cache_hit")) for r in rows),
        "failed_queries": sum(bool(r.get("failure")) for r in rows),
    }


def cluster_interval(rows, metric, *, seed=1729, replicates=2000):
    groups = {}
    for r in rows:
        groups.setdefault(r["group_id"], []).append(r)
    keys = sorted(groups)
    if not keys:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sample = [r for key in rng.choices(keys, k=len(keys)) for r in groups[key]]
        value = summary(sample)[metric]
        if value is not None and math.isfinite(value):
            values.append(value)
    return [percentile(values, 0.025), percentile(values, 0.975)] if values else None


def paired_comparison(base, candidate, *, seed=1729, replicates=2000):
    b = {r["id"]: r for r in base}
    c = {r["id"]: r for r in candidate}
    if b.keys() != c.keys() or len(b) != len(base) or len(c) != len(candidate):
        raise ValueError("Paired comparison requires identical unique question IDs")
    groups = {}
    for key in b:
        if b[key]["group_id"] != c[key]["group_id"]:
            raise ValueError("Group mismatch")
        groups.setdefault(b[key]["group_id"], []).append(key)
    latency_complete = all(
        summary(rows)["latency_seconds"]["p50"] is not None for rows in [base, candidate]
    )
    rng = random.Random(seed)
    keys = sorted(groups)
    values = {
        "accuracy_delta": [],
        "coverage_delta": [],
        "p50_latency_reduction": [],
        "p95_latency_reduction": [],
    }
    for _ in range(replicates):
        ids = [i for g in rng.choices(keys, k=len(keys)) for i in groups[g]]
        bs = summary([b[i] for i in ids])
        cs = summary([c[i] for i in ids])
        values["accuracy_delta"].append(cs["accuracy_all_decisions"] - bs["accuracy_all_decisions"])
        values["coverage_delta"].append(cs["definitive_coverage"] - bs["definitive_coverage"])
        for q in ["p50", "p95"] if latency_complete else []:
            denominator = bs["latency_seconds"][q]
            values[q + "_latency_reduction"].append(
                1 - cs["latency_seconds"][q] / denominator if denominator else 0
            )
    return {
        key: {
            "mean": statistics.mean(v),
            "cluster_bootstrap_95pct_interval": [percentile(v, 0.025), percentile(v, 0.975)],
        }
        if v
        else None
        for key, v in values.items()
    }
