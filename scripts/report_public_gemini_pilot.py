"""Reconcile bounded Cloud Logging exports without discarding failed or missing tasks."""

import argparse
import hashlib
import json
from pathlib import Path

from houndoracle.experiments.metrics import paired_comparison, summary

CAMPAIGN = "houndoracle-public-14"
MAXIMUM_RESERVATION = 0.021504


def reconcile(payload, logs):
    observed = {}
    versions = set()
    for entry in logs:
        event = entry.get("jsonPayload")
        if event is None and entry.get("textPayload"):
            try:
                event = json.loads(entry["textPayload"])
            except (ValueError, TypeError):
                continue
        if not isinstance(event, dict) or event.get("campaign") != CAMPAIGN:
            continue
        if event.get("kind") == "observed_version_pin":
            versions.add(event["version"])
        if event.get("kind") != "query":
            continue
        record = event["record"]
        key = (record["id"], record["repeat"])
        if key in observed and observed[key] != record:
            raise ValueError(f"Conflicting duplicate log record: {key}")
        observed[key] = record
    planned = {
        (r["id"], repeat): r
        for r in payload["questions"]
        for repeat in range(payload["repetitions"])
    }
    if observed.keys() - planned.keys():
        raise ValueError("Logs contain tasks outside the frozen workload")
    rows = []
    for key, source in planned.items():
        envelope = observed.get(key, {})
        result = dict(envelope.get("result") or {})
        if isinstance(result.get("answer"), dict):
            answer = result["answer"]
            result["answer"] = {
                k: answer[k] for k in ("decision", "evidence_indices") if k in answer
            }
            result["answer"]["citation_metadata"] = [
                {
                    "url": citation.get("url"),
                    "quote_sha256": hashlib.sha256(citation.get("quote", "").encode()).hexdigest(),
                    "quote_characters": len(citation.get("quote", "")),
                }
                for citation in answer.get("citations", [])
            ]
        if envelope and any(envelope[k] != source[k] for k in ("group_id", "expected")):
            raise ValueError("Logged labels or groups differ from frozen data")
        failure = result.get("failure")
        if envelope.get("state") != "SUCCEEDED":
            failure = failure or envelope.get("state", "UNOBSERVED")
        if result.get("reported_model_version"):
            versions.add(result["reported_model_version"])
        rows.append(
            {k: source[k] for k in ("id", "group_id", "expected")}
            | result
            | {
                "repeat": key[1],
                "decision": result.get("decision", "MAYBE"),
                "failure": failure,
                "latency_seconds": result.get("latency_seconds"),
                "all_in_cost_usd": None,
                "citation_supported": None,
                "execution_state": envelope.get("state", "UNOBSERVED"),
            }
        )
    return rows, sorted(versions)


def report(payload, logs, local, *, replicates=2000):
    remote, versions = reconcile(payload, logs)
    local_keys = {(r["id"], r["repeat"]) for r in local}
    remote_keys = {(r["id"], r["repeat"]) for r in remote}
    if local_keys != remote_keys or len(local_keys) != len(local):
        raise ValueError("Local trace is not the exact paired workload")
    # Distinct repetition IDs preserve all calls; group_id keeps repeated questions clustered.
    paired = []
    for rows in (remote, local):
        paired.append([r | {"id": f"{r['id']}:{r['repeat']}"} for r in rows])
    known = [r.get("token_cost_at_list_price_usd") for r in remote]
    observed_cost = sum(c for c in known if c is not None)
    unknown = sum(c is None for c in known)
    result = {
        "status": "OBSERVED_LOGS_NOT_INVOICE",
        "unique_questions": len(payload["questions"]),
        "repetitions": payload["repetitions"],
        "reported_model_versions": versions,
        "configuration_consistent": len(versions) == 1
        and versions[0].startswith("gemini-3.8-flash"),
        "gemini_matched_evidence": summary(remote),
        "local_calibrated": summary(local),
        "local_raw_candidate": summary([r | {"decision": r["candidate"]} for r in local]),
        "local_minus_gemini_paired_cluster_bootstrap": paired_comparison(
            *paired, seed=1930, replicates=replicates
        ),
        "gemini_token_list_price_usd_bounds": [
            observed_cost,
            observed_cost + unknown * MAXIMUM_RESERVATION,
        ],
        "unknown_token_cost_tasks": unknown,
        "cost_limitations": "No invoice, host/storage/network or local host cost measurement; implicit cached-token discounts not deducted. Unknown/unobserved tasks retain maximum reservations, including tasks that may never have executed. No all-in savings claim.",
        "scope": "Conditional historical claim-evidence labels; no live retrieval, citation audit or full agent/hybrid workflow. Local warm-model and remote API timing are distinct deployment measurements. Failed MAYBE is incorrect even when gold is MAYBE. Missing latency stays unknown. No locked-data tuning or model promotion follows this pilot.",
    }
    return remote, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path)
    args = parser.parse_args()
    payload = json.loads(Path("/workspace/houndoracle-data/public-gemini-payload.json").read_text())
    target = Path("benchmarks/results/campaign-14")
    if (target / "comparison.json").exists():
        raise FileExistsError("Comparison already frozen; do not overwrite")
    local = [json.loads(line) for line in (target / "local-records.jsonl").read_text().splitlines()]
    records, result = report(payload, json.loads(args.logs.read_text()), local)
    (target / "gemini-records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (target / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
