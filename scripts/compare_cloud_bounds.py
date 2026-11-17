"""Conservative uncertainty propagation for failed-call usage and missing timers."""

import json
import math
import random
from pathlib import Path


def quantile(values, q):
    a = sorted(values)
    index = (len(a) - 1) * q
    lo = int(index)
    hi = min(len(a) - 1, lo + 1)
    w = index - lo
    if w == 0:
        return a[lo]
    if not math.isfinite(a[lo]):
        return a[lo]
    if not math.isfinite(a[hi]):
        return a[hi]
    return a[lo] * (1 - w) + a[hi] * w


def bounds(rows, metric):
    if metric == "token_cost":
        low = sum(r["token_cost_usd"] or 0 for r in rows)
        upper = low + sum(
            r.get("reserved_usd", 0.021504) for r in rows if r["token_cost_usd"] is None
        )
        return low, upper
    q = {"p50": 0.5, "p95": 0.95}[metric]
    known = [r["latency_seconds"] for r in rows if r["latency_seconds"] is not None]
    missing = len(rows) - len(known)
    return quantile(known + [0.0] * missing, q), quantile(known + [float("inf")] * missing, q)


def reduction(base, candidate, metric):
    bl, bu = bounds(base, metric)
    cl, cu = bounds(candidate, metric)
    return (1 - cu / bl if bl > 0 else float("-inf"), 1 - cl / bu if bu > 0 else float("inf"))


def finite(x):
    return x if math.isfinite(x) else None


def main():
    report = json.loads(Path("/workspace/houndoracle-data/cloud-05-analysis.json").read_text())
    groups = {arm: {} for arm in ["baseline", "selected"]}
    for r in report["records"]:
        groups[r["arm"]].setdefault(r["id"], []).append(r)
    ids = sorted(groups["baseline"])
    rng = random.Random(1909)
    assert set(ids) == set(groups["selected"])
    draws = {m: [[], []] for m in ["token_cost", "p50", "p95"]}
    full = {a: [r for rows in g.values() for r in rows] for a, g in groups.items()}
    for _ in range(5000):
        selected = rng.choices(ids, k=len(ids))
        sample = {a: [r for ident in selected for r in groups[a][ident]] for a in groups}
        for m in draws:
            lower, upper = reduction(sample["baseline"], sample["selected"], m)
            draws[m][0].append(lower)
            draws[m][1].append(upper)
    result = {
        m: {
            "observed_reduction_identification_bounds": [
                finite(x) for x in reduction(full["baseline"], full["selected"], m)
            ],
            "paired_cluster_bootstrap_outer_95pct_interval": [
                finite(quantile(draws[m][0], 0.025)),
                finite(quantile(draws[m][1], 0.975)),
            ],
        }
        for m in draws
    }
    output = {
        "metrics": result,
        "question_clusters": len(ids),
        "bootstrap_replicates": 5000,
        "seed": 1909,
        "assumptions": [
            "Independent/exchangeable residual source components under this historical protocol.",
            "Missing per-query duration is nonnegative with no assumed finite upper bound.",
            "Unreported token usage consumes zero to its full pre-call reservation; no refunds assumed.",
            "Outer bootstrap diagnostic incorporates missing-value bounds; not a formal guarantee under distribution shift or optional stopping.",
        ],
        "limitations": [
            "The planned three-repeat experiment is incomplete; results are provisional, not a release gate pass.",
            "Token cost excludes compute/network/storage/logging, invoiced costs and discounts.",
            "Whole-population/all-category advantage is not established.",
        ],
    }
    Path("benchmarks/results/campaign-05/bounded-comparison.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
