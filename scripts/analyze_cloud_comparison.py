"""Analyze actual Cloud Logging records; never fabricates missing or failed calls."""

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy.stats import beta

from houndoracle.experiments.metrics import percentile


def events(raw, campaign="oracle-cloud-03"):
    result = []
    for item in raw:
        value = item.get("jsonPayload", item)
        if value.get("campaign") == campaign:
            result.append(value)
    return result


def audit(record, example):
    if record.get("failure"):
        return {
            "kind": "failed_or_unexecuted_task",
            "quotes_checked": 0,
            "note": "Failure is not evidence and does not establish a verdict.",
        }
    gemini = record.get("gemini")
    if not gemini:
        return {
            "kind": "oracle-original-evidence",
            "quotes_checked": len(example["evidence"]),
            "note": "Uses original supplied evidence; independent label consistency assessed separately.",
        }
    citations = gemini.get("answer", {}).get("citations", [])
    checks = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        quote = citation.get("quote") or citation.get("text") or ""
        url = citation.get("url")
        matched = any(
            e["url"] == url and bool(quote) and quote in e["text"] for e in example["evidence"]
        )
        checks.append(matched)
    return {
        "kind": "exact quote and URL presence",
        "citations": len(checks),
        "all_quotes_present": bool(checks) and all(checks),
        "note": "Exact quote presence alone does not establish entailment.",
    }


def summarize(rows):
    definitive = [r for r in rows if r["decision"] in ("YES", "NO") and not r.get("failure")]
    correct = lambda r: r["decision"] == r["expected"] and not r.get("failure")
    costs = [r.get("token_cost_usd") for r in rows]
    latencies = [r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None]
    missing_latencies = len(rows) - len(latencies)

    def bounded_quantile(q, upper=False):
        values = sorted(latencies + ([float("inf")] if upper else [0.0]) * missing_latencies)
        index = (len(values) - 1) * q
        lo, hi = int(index), min(len(values) - 1, int(index) + 1)
        weight = index - lo
        result = values[lo] if weight == 0 else values[lo] * (1 - weight) + values[hi] * weight
        return result if math.isfinite(result) else None

    known_cost = sum(c for c in costs if c is not None)
    unknown_cost_rows = [r for r in rows if r.get("token_cost_usd") is None]
    unknown_reservations = [r.get("reserved_usd") for r in unknown_cost_rows]
    unique = {}
    for row in rows:
        unique.setdefault(row["id"], []).append(row)
    successes = sum(all(correct(r) for r in group) for group in unique.values())
    n = len(unique)
    # Conservative repeated-question event, avoiding a zero-width interval when all outputs match.
    interval = (
        [
            float(beta.ppf(0.025, successes, n - successes + 1)) if successes else 0,
            float(beta.ppf(0.975, successes + 1, n - successes)) if successes < n else 1,
        ]
        if n
        else None
    )
    return {
        "task_executions": len(rows),
        "unique_questions": n,
        "accuracy": sum(correct(r) for r in rows) / len(rows),
        "definitive_coverage": len(definitive) / len(rows),
        "correct_definitive_fraction": sum(correct(r) for r in definitive) / len(rows),
        "accepted_accuracy": sum(correct(r) for r in definitive) / len(definitive)
        if definitive
        else None,
        "question_all_repetitions_correct_95pct_interval": interval,
        "p50_seconds": statistics.median(latencies) if not missing_latencies else None,
        "p95_seconds": percentile(latencies, 0.95) if not missing_latencies else None,
        "p99_seconds": percentile(latencies, 0.99) if not missing_latencies else None,
        "unknown_latency_count": missing_latencies,
        "latency_quantile_bounds_seconds": {
            name: [bounded_quantile(q), bounded_quantile(q, True)]
            for name, q in [("p50", 0.5), ("p95", 0.95), ("p99", 0.99)]
        },
        "known_reported_token_cost_usd": known_cost,
        "unreported_token_cost_calls": len(unknown_cost_rows),
        "token_cost_total_bounds_usd": [
            known_cost,
            known_cost + sum(unknown_reservations)
            if all(x is not None for x in unknown_reservations)
            else None,
        ],
        "token_list_price_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
        "model_calls": sum(r["model_calls"] for r in rows),
        "failed_queries": sum(bool(r.get("failure")) for r in rows),
        "reported_input_tokens": sum(
            (r.get("gemini") or {}).get("input_tokens") or 0 for r in rows
        ),
        "reported_visible_output_tokens": sum(
            (r.get("gemini") or {}).get("output_tokens") or 0 for r in rows
        ),
        "reported_reasoning_tokens": sum(
            (r.get("gemini") or {}).get("reasoning_tokens") or 0 for r in rows
        ),
        "model_calls_without_reported_reasoning_count": sum(
            r.get("gemini") is not None and r["gemini"].get("reasoning_tokens") is None
            for r in rows
        ),
        "actual_invoiced_all_in_cost_usd": None,
    }


def analyze(raw, payload, campaign="oracle-cloud-03"):
    logs = events(raw, campaign)
    examples = {r["id"]: r for part in ["development", "confirmation"] for r in payload[part]}
    queries = []
    for item in logs:
        if item["kind"] != "query":
            continue
        row = item["record"]
        example = examples[row["id"]]
        if row["expected"] != example["expected"]:
            raise ValueError("Gold label changed in remote record")
        queries.append({**row, "citation_audit": audit(row, example)})
    keys = [(r["phase"], r["arm"], r["id"], r["repeat"]) for r in queries]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate log records; deduplicate by insertId before analysis")
    groups = {}
    for row in queries:
        groups.setdefault(row["phase"] + ":" + row["arm"], []).append(row)
    summaries = {k: summarize(v) for k, v in groups.items()}
    models = [r for r in logs if r["kind"] == "model_selected"]
    complete = any(r["kind"] == "completed" for r in logs)
    usage_audit = []
    calls = [r["gemini"] for r in queries if r.get("gemini")]
    calls += [r["smoke"] for r in logs if r["kind"] == "model_selected"]
    for call in calls:
        usage = call.get("raw_response", {}).get("usage_metadata", {})
        total = usage.get("total_token_count")
        known = sum(
            usage.get(k, 0) or 0
            for k in ["prompt_token_count", "candidates_token_count", "thoughts_token_count"]
        )
        usage_audit.append(
            {
                "total": total,
                "accounted": known,
                "residual": total - known if total is not None else None,
            }
        )
    repetitions = payload.get("repetitions", 3 if campaign == "oracle-cloud-03b" else 2)
    expected_count = 2 * repetitions * (len(payload["development"]) + len(payload["confirmation"]))
    complete = (
        complete
        and len(queries) == expected_count
        and not any(r.get("failure") == "NOT_EXECUTED" for r in queries)
    )
    report = {
        "campaign": campaign,
        "expected_task_records": expected_count,
        "token_accounting_audit": usage_audit,
        "complete": complete,
        "models": models,
        "systems": summaries,
        "selection": [r for r in logs if r["kind"] == "selection"],
        "configuration": [r for r in logs if r["kind"] == "configuration"],
        "unavailable_models": [r for r in logs if r["kind"] == "model_unavailable"],
        "limitations": [
            "Tiny pilot; confidence intervals are wide.",
            "Matched historical evidence, no Google Search grounding or autonomous agent tool loop.",
            "Model list-price usage is not verified all-in billing; startup/compute/egress/logging remain separate.",
            "Incomplete execution must not be reported as a completed comparison.",
        ],
        "records": queries,
    }
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logs", required=True)
    p.add_argument("--payload", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--campaign", default="oracle-cloud-03")
    args = p.parse_args()
    result = analyze(
        json.loads(Path(args.logs).read_text()),
        json.loads(Path(args.payload).read_text()),
        args.campaign,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["records", "models"]}, indent=2))
