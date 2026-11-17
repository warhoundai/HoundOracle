"""Reconcile integration, rejected training and research calibration without promotion."""

import hashlib
import json
from pathlib import Path

from houndoracle.experiments.fit_calibration import diagnostics
from houndoracle.experiments.metrics import paired_comparison, summary


def main():
    root = Path("research/experiments")
    private = Path("/workspace/houndoracle-data")
    result = json.loads((private / "campaign-46/result.json").read_text())
    inputs = json.loads((private / "campaign-46/payload.json").read_text())
    original = [(r["id"], r["group_id"], r["expected"]) for r in inputs["development"]]
    for arm in ["baseline", "candidate"]:
        if [(r["id"], r["group_id"], r["expected"]) for r in result[arm]] != original:
            raise ValueError("Incomplete/misaligned500case observations")
    if (
        hashlib.file_digest(
            (private / "campaign-46/adapter.safetensors").open("rb"), "sha256"
        ).hexdigest()
        != result["adapter_sha256"]
    ):
        raise ValueError("Adapter changed")
    arms = {
        k: [
            r | {"decision": r["candidate"], "citation_supported": None, "all_in_cost_usd": None}
            for r in result[k]
        ]
        for k in ["baseline", "candidate"]
    }
    stats = {k: summary(v) for k, v in arms.items()}
    passed = (
        result["training_complete"]
        and stats["candidate"]["accuracy_all_decisions"]
        >= stats["baseline"]["accuracy_all_decisions"] + 8 / 500 - 1e-12
        and stats["candidate"]["definitive_coverage"]
        >= stats["baseline"]["definitive_coverage"] - 0.02
    )
    assessment = {
        "systems_unqualified_candidates": stats,
        "paired": paired_comparison(
            arms["baseline"], arms["candidate"], replicates=2000, seed=2005
        ),
        "training_steps": result["training_steps"],
        "training_complete": result["training_complete"],
        "training_seconds": result["training_seconds"],
        "total_application_seconds": result["total_seconds"],
        "adapter_sha256": result["adapter_sha256"],
        "decision": "retain_for_further_validation" if passed else "reject",
        "screen_passed": passed,
        "actual_invoice_usd": None,
        "reservation_usd": 12,
        "production_qualified": False,
        "scope": "New500case development, original human annotations but independent source/citation audits pending. No97%or95%acceptance gate passed. Baseline-before-training/candidate-after-training timings are not randomized paired agent latency comparisons.",
    }
    (root / "campaign-46/screen.json").write_text(json.dumps(assessment, indent=2) + "\n")
    (root / "campaign-46/training-curve.json").write_text(
        json.dumps(result["curve"], indent=2) + "\n"
    )
    out = Path("benchmarks/results/campaign-46")
    out.mkdir(parents=True, exist_ok=True)
    for k, rows in arms.items():
        (out / (k + "-metadata.jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
    old = [
        json.loads(l)
        for l in Path("benchmarks/results/campaign-44-reproduction/general-metadata.jsonl")
        .read_text()
        .splitlines()
    ]
    new = [
        json.loads(l)
        for l in Path("benchmarks/results/campaign-45/general-metadata.jsonl")
        .read_text()
        .splitlines()
    ]
    resolution = {}
    for name, rows in [("v1_reproduced", old), ("v2", new)]:
        definitive = [r for r in rows if r["candidate"] in ["YES", "NO"]]
        resolution[name] = {
            "count": len(rows),
            "correct_candidates": sum(r["candidate"] == r["expected"] for r in rows),
            "distinct_definitive_scores": len({r["raw_score"] for r in definitive}),
            "uncalibrated_correctness_brier": sum(
                (r["raw_score"] - int(r["candidate"] == r["expected"])) ** 2 for r in definitive
            )
            / len(definitive),
        }
    (root / "campaign-45/score-resolution.json").write_text(json.dumps(resolution, indent=2) + "\n")
    validation = [
        json.loads(l)
        for l in (private / "campaign-47/validation-scores.jsonl").read_text().splitlines()
    ]
    mapping = json.loads((root / "campaign-47/frozen-map.json").read_text())["maps"]["logistic"]
    cal = {"raw": diagnostics(validation), "mapped": diagnostics(validation, mapping)}
    (root / "campaign-47/calibration-diagnostics.json").write_text(json.dumps(cal, indent=2) + "\n")
    # Scientific export artifacts; threshold choices remain preregistered.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, data in cal.items():
        bins = data["reliability_bins"]
        axes[0].plot(
            [b["mean_confidence"] for b in bins],
            [b["empirical_correctness"] for b in bins],
            marker="o",
            label=name,
        )
        curve = [c for c in data["risk_coverage"] if c["selective_risk"] is not None]
        axes[1].plot(
            [c["coverage"] for c in curve],
            [c["selective_risk"] for c in curve],
            marker="o",
            label=name,
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(
        xlabel="Correctness score / mapped estimate",
        ylabel="Observed correctness",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Reliability on500validationcases",
    )
    axes[1].set(
        xlabel="Definitive coverage",
        ylabel="Selective risk",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Predeclared threshold sweep",
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Research only: supplied historical passages; source audits pending")
    out = Path("benchmarks/results/campaign-47")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "calibration.svg")
    fig.savefig(out / "calibration.png", dpi=160)
    plt.close(fig)
    report = """# General verification implementation and first optimization campaign

The95%target remains unmet. This campaign implements the public general-claim
path and acceptance machinery; it does not complete the independent end-to-end
evaluation, qualification or release phases.

## Implemented behavior

The public verifier can preserve exact prose propositions, evaluate qualified
source passages using a pinned local model and group evidence before acceptance.
It supports explicit outer negation without paraphrase-based answer reuse. Search
and full acquisition use existing self-hosted adapters and bounded budgets.
Machine acquisition is not human review. Exact spans, freshness, contradiction
search, failure reasons and per-stage timings are retained. Legacy deterministic
serving remains available without the semantic model.

A discovered long-document starvation bug was corrected with lexical passage
ranking and round-robin document coverage. Required counter-evidence must be
acquired and evaluated before acceptance. Revocation, cancellation and deadline
checks also apply to cached results. Changing model/aggregator/request policy
invalidates qualification. Latest controller policy isv4; frozenv3research
calibration remains archival and is not silently rebound to the new policy.

## Measured results

| Experiment | Result | Decision |
|---|---|---|
| Public API integration,207olddevelopment cases |176/207correct diagnostic candidates(85.02%); accepted definitive coverage0% | Keep integration; do not claim accepted accuracy improvement |
| Uncertainty-preserving aggregation | Same176correct;143distinct definitive scores instead of2; diagnostic Brier0.150738→0.102550 | Keep score-resolution fix; not a truth-probability guarantee |
| CPU adaptation,4096training +500newdevelopment cases | Baseline394/500(78.8%), adapted389/500(77.8%);512updates,407.61trainingseconds | Reject candidate; stable weights retained |
| Primary research calibration risk panel |223/244selected groups correct; conservative lower bound86.76% | Pass80%research risk check only |
| Separate500case validation | Raw candidates391/500(78.2%); research-selected precision226/244(92.62%), coverage48.8%, overall63.8% | Does not meet95%accuracy/coverage gates |

Accepted serving decisions still abstain without audited policy qualification.
Research mapping estimates are not objective truth probabilities or universal
future guarantees. Calibration improves how uncertainty is measured; it does not
repair the underlying model's refutation errors. The failed adaptation lost eight
correct NO cases, one correct YES case and gained four correct MAYBE cases.

## Data and audit readiness

A new candidate pool reserves500development,500validation,1000calibration and
1000locked passage cases, plus7816training candidates. Source/event components
are separated and prior research/training sources excluded. Existing39metadata
was read only to exclude related sources; its labels were not scored. New locked
labels were not scored or used for training. Publisher human annotations remain
unchanged. Source/citation/domain audits are pending, and the separate end-to-end
world-fact dataset is not yet qualified. Passage NEI cannot substitute for web
truth. The qualification CLI reports blocked rather than fabricating a sealed set.

Source-specific Wikipedia/CC-BY-SA terms were inspected for private research.
No dataset or trained model public redistribution is authorized by that review.
FEVER source-component preparation failures and the empty-identifier repair are
preserved in the qualification directory. No source family was chosen based on
model evaluation outcomes.

## Failures, reproducibility and resources

An exporter path bug overwrote original44per-case traces. Original aggregates
remain. Actual second-run traces were moved to45; archived44source was hash
checked and reproduced separately. Reproduced timings are new observations,
not recovered original timings. The incident and all available records remain.

The temporary8vCPU EC2 worker was terminated and its role, profile and security
group deleted after collection. Price verified:$0.357/hour. Application runtime
was501.57seconds, excluding bootstrap; invoices and complete operational cost
remain unknown.12USD was reserved, not claimed as actual spending. AWS campaign
reservations now84USD/unreserved216USD; GCP26USD/unreserved24USD unchanged.
No paid-model calls, public endpoint, OPS change or public publication occurred.

The source pool, trained safetensors adapter, fitting/risk/validation records,
training curve, calibration maps, reliability plot and per-case metadata are
retained. The strongest previously validated baseline checkpoint is unchanged.

## Next acceptance gates

Complete independent source/citation/domain audits and end-to-end labels. Improve
refutation and multi-hop evidence understanding on development cases. Freeze the
serving policy, validate both tracks at97%, recalibrate that exact policy and only
then open the two1000case tests. Both must reach950correct, with precision,
answerable coverage and citation correctness at95%and zero critical regressions.
"""
    # Improve readability without changing identifiers or measurements.
    import re

    report = re.sub(
        r"(?<=[A-Za-z])(95|97|207|176|143|223|244|4096|500|512|1000|7816|501|12|84|216|26|24|950|96)(?=[%A-Za-z])",
        r" \1",
        report,
    )
    report = re.sub(
        r"(207|176|143|223|244|4096|500|512|1000|7816|950|95|97)(?=[A-Za-z])", r"\1 ", report
    )
    Path("research/general-engineering-report-44-47.md").write_text(report)
    print(
        json.dumps(
            {"training_candidate_promoted": passed, "qualified95": False, "plots_written": True}
        )
    )


if __name__ == "__main__":
    main()
