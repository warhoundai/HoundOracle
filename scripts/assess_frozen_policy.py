"""One-shot independent risk qualification, followed by untouched tests; no threshold retuning."""

import hashlib
import json
import random
from pathlib import Path

from scipy.stats import beta

from houndoracle.experiments.abstention import would_abstain
from houndoracle.experiments.head import HeadNLI
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.pipeline import ResearchPipeline


def upper(k, n, alpha=0.05):
    return float(beta.ppf(1 - alpha, k + 1, n - k)) if n and k < n else 1.0


def main():
    output = Path("research/experiments/campaign-06")
    result_path = output / "independent-risk.json"
    if result_path.exists():
        raise FileExistsError("Fresh risk already assessed; do not rerun/tune it")
    policy = json.loads((output / "final-policy-frozen.json").read_text())
    if not policy["validation_policy_passed"]:
        raise ValueError("Candidate failed validation; preserve fresh data")
    newcal = json.loads((output / "calibration-finetuned.json").read_text())
    oldcal = json.loads(
        Path("research/experiments/campaign-04/calibration-logistic-threads2.json").read_text()
    )
    head = json.loads((output / "finetuned-head.json").read_text())
    oldhead = json.loads(Path("research/experiments/results/trained-aggregators.json").read_text())[
        "logistic"
    ]
    manifest = json.loads(
        Path("research/model-comparisons/finetuned-minilm-provenance.json").read_text()
    )
    models = {
        "baseline": ResearchPipeline(
            HeadNLI(
                OnnxNLI("/workspace/houndoracle-models/minilm", variant="fp32", threads=2),
                oldhead,
                "logistic",
            ),
            oldcal,
        ),
        "finetuned": ResearchPipeline(
            HeadNLI(
                OnnxNLI(
                    "/workspace/houndoracle-models/minilm-fever-1912",
                    variant="fp32",
                    threads=2,
                    reviewed_manifest=manifest,
                ),
                head,
                "finetuned-logistic",
            ),
            newcal,
        ),
    }
    safe = Path("benchmarks/results/campaign-06")
    private = Path("/workspace/houndoracle-data/campaign-06-assessment")
    private.mkdir(exist_ok=True)
    rng = random.Random(1914)

    def score(path, panel, stage):
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rng.shuffle(rows)
        records = []
        with (private / f"{panel}-{stage}.jsonl").open("w") as f:
            for i, row in enumerate(rows):
                order = list(models)
                rng.shuffle(order)
                for name in order:
                    model = models[name]
                    result = model.verify_evidence(
                        row["question"],
                        row["evidence"],
                        population=model.artifact["population"],
                        use_cache=False,
                    )
                    threshold = 0.85 if name == "baseline" else policy["minimum_confidence"]
                    accepted = (
                        result["decision"] in ["YES", "NO"]
                        and (result["confidence"] or 0) >= threshold
                    )
                    gate = (
                        name == "finetuned"
                        and would_abstain(result, policy["insufficiency_gate"])
                        and policy["insufficiency_gate_screen_passed"]
                    )
                    item = {
                        "id": row["id"],
                        "group_id": row["group_id"],
                        "expected": row["expected"],
                        "panel": panel,
                        "stage": stage,
                        "model": name,
                        "decision": result["decision"] if accepted else "MAYBE",
                        "local_definitive": accepted,
                        "proposed_local_abstention": bool(gate),
                        "confidence": result["confidence"] if accepted else None,
                        "latency_seconds": result["latency_seconds"],
                        "raw_score": result.get("raw_score"),
                    }
                    records.append(item)
                    f.write(json.dumps(item) + "\n")
                if (i + 1) % 200 == 0:
                    print(
                        json.dumps({"panel": panel, "stage": stage, "questions": i + 1}), flush=True
                    )
        return records

    balanced = score(
        Path("/workspace/houndoracle-data/campaign-06-balanced/risk-extended.jsonl"),
        "balanced",
        "risk",
    )
    strict = score(
        Path("/workspace/houndoracle-data/campaign-06/risk.jsonl"), "strict-pages", "risk"
    )

    def summarize(rows):
        answers = [r for r in rows if r["local_definitive"]]
        wrong = sum(r["decision"] != r["expected"] for r in answers)
        gates = [r for r in rows if r["proposed_local_abstention"]]
        return {
            "rows": len(rows),
            "local_definitive_coverage": len(answers) / len(rows),
            "correct_definitive_fraction": sum(r["decision"] == r["expected"] for r in answers)
            / len(rows),
            "selective_error": wrong / len(answers) if answers else None,
            "local_definitive_errors": wrong,
            "local_definitive_count": len(answers),
            "selective_error_cp_upper": upper(wrong, len(answers)),
            "proposed_abstention_fraction": len(gates) / len(rows),
            "false_abstentions": sum(r["expected"] != "MAYBE" for r in gates),
            "local_verifier_accuracy": sum(r["decision"] == r["expected"] for r in rows)
            / len(rows),
            "note": "Gemini fallback not executed; local-verifier accuracy is not hybrid workflow accuracy.",
        }

    by_group = {
        (panel, name): [r for r in records if r["model"] == name]
        for panel, records in [("balanced", balanced), ("strict-pages", strict)]
        for name in models
    }
    stats = {panel + ":" + name: summarize(rows) for (panel, name), rows in by_group.items()}
    new = by_group["balanced", "finetuned"]
    bounds = {}
    for label in ["YES", "NO"]:
        cohort = [r for r in new if r["expected"] == label]
        k = sum(r["proposed_local_abstention"] for r in cohort)
        bounds[label] = {
            "errors": k,
            "rows": len(cohort),
            "one_sided_97_5pct_upper": upper(k, len(cohort), 0.025),
        }
    balanced_loss_upper = sum(v["one_sided_97_5pct_upper"] for v in bounds.values()) / 3
    natural = by_group["strict-pages", "finetuned"]
    natural_k = sum(r["proposed_local_abstention"] and r["expected"] != "MAYBE" for r in natural)
    natural_loss_upper = upper(natural_k, len(natural))
    truth_qualified = (
        stats["balanced:finetuned"]["local_definitive_count"] >= 30
        and stats["balanced:finetuned"]["selective_error_cp_upper"] <= 0.2
    )
    gate_qualified = (
        truth_qualified
        and policy["insufficiency_gate_screen_passed"]
        and balanced_loss_upper <= 0.02
        and natural_loss_upper <= 0.02
        and stats["balanced:finetuned"]["proposed_abstention_fraction"] >= 0.05
        and stats["strict-pages:finetuned"]["proposed_abstention_fraction"] >= 0.05
    )
    report = {
        "statistics": stats,
        "truth_policy_qualified_for_research": truth_qualified,
        "insufficiency_gate_qualified": gate_qualified,
        "stratum_loss_bounds": bounds,
        "balanced_opportunity_loss_upper": balanced_loss_upper,
        "strict_page_opportunity_loss_upper": natural_loss_upper,
        "assumptions": "Component exchangeability within declared residual datasets; balanced stratum CP diagnostics use Bonferroni alpha .025. Not a shift-proof guarantee or general production confidence.",
        "frozen_policy_sha256": hashlib.sha256(
            (output / "final-policy-frozen.json").read_bytes()
        ).hexdigest(),
        "production_qualified": False,
    }
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    deployment = dict(policy)
    deployment["insufficiency_gate"]["risk_certificate"] = {
        "qualified": gate_qualified,
        "balanced_opportunity_loss_upper": balanced_loss_upper,
        "strict_page_opportunity_loss_upper": natural_loss_upper,
    }
    deployment["independent_truth_risk_passed"] = truth_qualified
    (output / "assessed-policy.json").write_text(json.dumps(deployment, indent=2) + "\n")
    final = []
    if truth_qualified:
        final += score(
            Path("/workspace/houndoracle-data/campaign-06-balanced/locked.jsonl"),
            "balanced",
            "locked",
        )
        final += score(
            Path("/workspace/houndoracle-data/campaign-06/locked.jsonl"), "strict-pages", "locked"
        )
    with (safe / "independent-records.jsonl").open("w") as f:
        for r in balanced + strict + final:
            f.write(json.dumps(r) + "\n")
    summary = {
        "risk": report,
        "locked": {
            panel + ":" + name: summarize(
                [r for r in final if r["panel"] == panel and r["model"] == name]
            )
            for panel in ["balanced", "strict-pages"]
            for name in models
        }
        if final
        else {},
        "heldout_scored_once": bool(final),
        "new_provider_calls": 0,
        "all_in_cost_usd": None,
        "decision": "retain-scoped-neural-research-policy"
        if truth_qualified
        else "reject-before-locked-test",
    }
    (safe / "independent-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
