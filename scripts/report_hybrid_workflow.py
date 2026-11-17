"""Complete planned-slot analysis for actual colocated Oracle/Vertex fallback runs."""

import argparse
import json
from pathlib import Path

from houndoracle.experiments.metrics import paired_comparison, summary


def reconcile(payload, logs):
    seen = {}
    for entry in logs:
        event = entry.get("jsonPayload")
        if event is None and entry.get("textPayload"):
            try:
                event = json.loads(entry["textPayload"])
            except (TypeError, ValueError):
                continue
        if (
            not isinstance(event, dict)
            or event.get("campaign") != "houndoracle-workflow-18"
            or event.get("kind") != "query"
        ):
            continue
        record = event["record"]
        key = (record["id"], record["repeat"], record["arm"])
        if key in seen and seen[key] != record:
            raise ValueError("Conflicting duplicate workflow record")
        seen[key] = record
    planned = {
        (r["id"], rep, arm): r
        for r in payload["questions"]
        for rep in range(payload["repetitions"])
        for arm in ["gemini", "hybrid"]
    }
    if seen.keys() - planned.keys():
        raise ValueError("Unexpected task outside frozen plan")
    rows = []
    for (ident, rep, arm), source in planned.items():
        envelope = seen.get((ident, rep, arm), {})
        result = envelope.get("result") or {}
        if envelope and any(envelope[k] != source[k] for k in ["group_id", "expected"]):
            raise ValueError("Gold/group mismatch")
        failure = result.get("failure")
        if envelope.get("state") != "SUCCEEDED":
            failure = failure or envelope.get("state", "UNOBSERVED")
        local = result.get("local") or {}
        rows.append(
            {k: source[k] for k in ["id", "group_id", "expected"]}
            | {
                "repeat": rep,
                "arm": arm,
                "decision": result.get("decision", "MAYBE"),
                "failure": failure,
                "source": result.get("source"),
                "execution_state": envelope.get("state", "UNOBSERVED"),
                "latency_seconds": envelope.get("workflow_latency_seconds")
                if envelope.get("executed")
                else None,
                "model_api_seconds": result.get("latency_seconds"),
                "local_cache_hit": local.get("cache_hit"),
                "local_seconds": local.get("latency_seconds"),
                "model_calls": result.get("model_calls"),
                "input_tokens": result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
                "reasoning_tokens": result.get("reasoning_tokens"),
                "reported_model_version": result.get("reported_model_version"),
                "token_cost_at_list_price_usd": result.get("token_cost_at_list_price_usd"),
                "all_in_cost_usd": None,
                "citation_supported": None,
            }
        )
    return rows


def report(payload, logs, *, replicates=2000):
    rows = reconcile(payload, logs)
    arms = {arm: [r for r in rows if r["arm"] == arm] for arm in ["gemini", "hybrid"]}
    summaries = {arm: summary(rs) for arm, rs in arms.items()}
    costs = {}
    for arm, rs in arms.items():
        known = sum(
            r["token_cost_at_list_price_usd"]
            for r in rs
            if r["token_cost_at_list_price_usd"] is not None
        )
        unknown = sum(r["token_cost_at_list_price_usd"] is None for r in rs)
        costs[arm] = {
            "known_reported_token_cost_usd": known,
            "unknown_cost_slots": unknown,
            "token_list_price_bounds_usd": [known, known + unknown * 0.021504],
            "observed_model_calls": sum(r["model_calls"] or 0 for r in rs),
        }
    paired = paired_comparison(
        *[
            [r | {"id": f"{r['id']}:{r['repeat']}"} for r in arms[arm]]
            for arm in ["gemini", "hybrid"]
        ],
        seed=1934,
        replicates=replicates,
    )
    base = summaries["gemini"]
    hybrid = summaries["hybrid"]
    reductions = {
        q: 1 - hybrid["latency_seconds"][q] / base["latency_seconds"][q]
        if hybrid["latency_seconds"][q] is not None and base["latency_seconds"][q]
        else None
        for q in ["p50", "p95"]
    }
    cost_reduction = (
        1
        - costs["hybrid"]["known_reported_token_cost_usd"]
        / costs["gemini"]["known_reported_token_cost_usd"]
        if not any(v["unknown_cost_slots"] for v in costs.values())
        and costs["gemini"]["known_reported_token_cost_usd"]
        else None
    )
    output = {
        "status": "OBSERVED_WORKFLOW_NOT_INVOICE",
        "summaries": summaries,
        "model_token_costs": costs,
        "observed_token_cost_reduction": cost_reduction,
        "observed_latency_reductions": reductions,
        "paired_cluster_bootstrap": paired,
        "by_result_cache_temperature": {
            ("cold" if rep == 0 else "warm-" + str(rep)): {
                arm: summary([r for r in rs if r["repeat"] == rep]) for arm, rs in arms.items()
            }
            for rep in range(payload["repetitions"])
        },
        "oracle_model_requests_avoided": sum(
            r["source"] == "oracle" and not r["failure"] for r in arms["hybrid"]
        ),
        "hybrid_local_cache_hits": sum(r["local_cache_hit"] is True for r in arms["hybrid"]),
        "quality_noninferiority_statistically_established": False,
        "public_release_qualified": False,
        "scope": "Actual warm-encoder, colocated local-verifier/remote-model fallback workflow on historical matched evidence; reused48-source cohort, not fresh independent web research. No initial generative agent/retrieval/citation-audit cost. Source-content and weights private. Oracle cache is local exact result reuse, not central public evidence economics. Failure slots retained and unknown latency not imputed.",
        "cost_limitations": "Token list prices exclude host, startup, storage, transfer, credits and invoice adjustments. Per-arm job-host allocation is not independently billed. No all-in cost claim.",
    }
    return rows, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path)
    args = parser.parse_args()
    target = Path("benchmarks/results/campaign-18")
    if (target / "comparison.json").exists():
        raise FileExistsError("Comparison already frozen")
    payload = json.loads(Path("/workspace/houndoracle-data/workflow18-payload.json").read_text())
    records, result = report(payload, json.loads(args.logs.read_text()))
    (target / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (target / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
