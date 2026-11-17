"""Export source-text-free measurements and paired question-cluster uncertainty."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from houndoracle.experiments.metrics import percentile


def compare(records, phase, candidate):
    arms = {
        a: [r for r in records if r["phase"] == phase and r["arm"] == a]
        for a in ["baseline", candidate]
    }
    ids = sorted({r["id"] for r in arms["baseline"]})
    if {r["id"] for r in arms[candidate]} != set(ids):
        raise ValueError("Unpaired question set")

    def measures(rows):
        return {
            "accuracy": sum(r["decision"] == r["expected"] and not r.get("failure") for r in rows)
            / len(rows),
            "coverage": sum(r["decision"] in {"YES", "NO"} and not r.get("failure") for r in rows)
            / len(rows),
            "cost": sum(r["token_cost_usd"] for r in rows) / len(rows)
            if all(r.get("token_cost_usd") is not None for r in rows)
            else None,
            "p50": percentile([r["latency_seconds"] for r in rows], 0.5)
            if all(r.get("latency_seconds") is not None for r in rows)
            else None,
            "p95": percentile([r["latency_seconds"] for r in rows], 0.95)
            if all(r.get("latency_seconds") is not None for r in rows)
            else None,
        }

    def differences(groups):
        b, c = measures(groups["baseline"]), measures(groups[candidate])
        return {k: c[k] - b[k] if c[k] is not None and b[k] is not None else None for k in b}

    observed = differences(arms)
    samples = {k: [] for k in observed}
    rng = random.Random(1905)
    for _ in range(10000):
        selected = rng.choices(ids, k=len(ids))
        draw = {
            a: [r for ident in selected for r in rows if r["id"] == ident]
            for a, rows in arms.items()
        }
        for key, value in differences(draw).items():
            if observed[key] is not None and value is not None:
                samples[key].append(value)
    return {
        "phase": phase,
        "candidate": candidate,
        "unique_questions": len(ids),
        "candidate_minus_baseline": observed,
        "paired_question_cluster_bootstrap_95pct_intervals": {
            k: [percentile(v, 0.025), percentile(v, 0.975)] if v else None
            for k, v in samples.items()
        },
        "bootstrap_seed": 1905,
        "bootstrap_draws": 10000,
        "caveat": f"{len(ids)} question clusters; empirical bootstrap may be unstable on small samples. Missing latency/cost values suppress point comparisons and intervals. Zero observed accuracy difference does not prove equivalence or generalization.",
    }


def export(analysis, destination):
    destination.mkdir(parents=True, exist_ok=True)
    records = analysis["records"]
    safe = []
    for r in records:
        gemini = r.get("gemini") or {}
        oracle = r.get("oracle") or {}
        safe.append(
            {
                k: r.get(k)
                for k in [
                    "id",
                    "group_id",
                    "phase",
                    "arm",
                    "repeat",
                    "expected",
                    "decision",
                    "source",
                    "latency_seconds",
                    "token_cost_usd",
                    "model_calls",
                    "failure",
                    "citation_audit",
                ]
            }
            | {
                "oracle_confidence": oracle.get("confidence"),
                "model_config": gemini.get("config"),
                "model_version": gemini.get("reported_model_version"),
                "input_tokens": gemini.get("input_tokens"),
                "output_tokens": gemini.get("output_tokens"),
                "reasoning_tokens": gemini.get("reasoning_tokens"),
                "raw_response_sha256": hashlib.sha256(
                    json.dumps(gemini.get("raw_response"), sort_keys=True).encode()
                ).hexdigest()
                if gemini
                else None,
                "actual_all_in_cost_usd": None,
                "reserved_usd": r.get("reserved_usd"),
                "latency_note": r.get("latency_note"),
                "record_origin": r.get("record_origin"),
                "execution_segment": r.get("execution_segment"),
                "source_attribution": "FEVER human labels / Wikipedia evidence; original text excluded; see research/reproduction.md and docs/rights.md",
            }
        )
    (destination / "per-question.jsonl").write_text("".join(json.dumps(r) + "\n" for r in safe))
    summary = {k: v for k, v in analysis.items() if k not in ["records", "models"]}
    summary["comparisons"] = [
        compare(records, phase, candidate)
        for phase, candidate in (
            [("supplemental", "selected")]
            if analysis["campaign"].endswith("03b")
            else (
                [("fresh", "selected")]
                if analysis["campaign"] == "oracle-cloud-05"
                else [("development", "mlp"), ("confirmation", "selected")]
            )
        )
    ]
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    export(json.loads(Path(args.analysis).read_text()), Path(args.output))
