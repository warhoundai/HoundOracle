"""One-shot independent locked-query assessment with fixed model selection and paired uncertainty."""

import hashlib
import json
from pathlib import Path

import numpy as np
from fit_query_risk import SPECS
from scipy.stats import beta, binomtest

from houndoracle.experiments.fit_calibration import diagnostics
from houndoracle.experiments.head import infer_head
from houndoracle.experiments.metrics import summary
from houndoracle.experiments.pipeline import apply


def ci(k, n, alpha=0.05):
    return [
        float(beta.ppf(alpha / 2, k, n - k + 1)) if k else 0.0,
        float(beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0,
    ]


def main():
    root = Path("research/experiments/campaign-09")
    safe = Path("benchmarks/results/campaign-09")
    target = safe / "locked-summary.json"
    if target.exists():
        raise FileExistsError("Locked results already assessed")
    gate = json.loads((root / "locked-authorization.json").read_text())
    if gate.get("qualified") is not True:
        raise PermissionError("No qualified fixed configuration")
    for name, sha in gate["file_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen model/dataset policy changed")
    expected = set(
        json.loads((root / "dataset-manifest.json").read_text())["panels"]["locked"]["ids"]
    )
    reports = {}
    all_rows = {}
    records = []
    for name, (model, _, _) in SPECS.items():
        private = Path("/workspace/houndoracle-data/campaign-09/scores") / model
        raw = [json.loads(s) for s in (private / "locked.jsonl").read_text().splitlines()]
        if (
            len(raw) != len(expected)
            or {r["id"] for r in raw} != expected
            or len({r["group_id"] for r in raw}) != len(raw)
        ):
            raise ValueError("Incomplete/dependent locked query records")
        head = json.loads((root / (name + "-head.json")).read_text())
        artifact = json.loads((root / (name + "-calibration.json")).read_text())
        if any(
            artifact["model_config"][k] != v
            for k, v in json.loads((private / "model.json").read_text()).items()
        ):
            raise ValueError("Encoder/calibration mismatch")
        predictions = [
            infer_head(r, head, evidence_group_count=r["evidence_group_count"]) for r in raw
        ]
        rows = [apply(r, artifact) for r in predictions]
        for r in rows:
            if r["decision"] in ["YES", "NO"]:
                assert (
                    r["confidence"] >= 0.8 and r["calibration_qualified"] and not r.get("truncated")
                )
        unqualified = [r | {"decision": r["candidate"]} for r in predictions]
        raw_summary, cal_summary = summary(unqualified), summary(rows)
        raw_summary["latency_seconds"] = None
        cal_summary["latency_seconds"] = None
        n = len(rows)
        accepted = [r for r in rows if r["decision"] in ["YES", "NO"]]
        k = sum(r["decision"] != r["expected"] for r in accepted)
        upper = (
            float(beta.ppf(1 - 0.05 / 5, k + 1, len(accepted) - k))
            if accepted and k < len(accepted)
            else 1.0
        )
        report = {
            "raw": raw_summary,
            "calibrated080": cal_summary,
            "raw_accuracy_95_exact_interval": ci(
                sum(r["decision"] == r["expected"] for r in unqualified), n
            ),
            "coverage_95_exact_interval": ci(len(accepted), n),
            "selected_errors": k,
            "selected_answers": len(accepted),
            "selective_risk_95_exact_interval": ci(k, len(accepted)) if accepted else None,
            "selective_risk_family_adjusted_upper": upper,
            "heldout_risk_gate_passed": bool(len(accepted) >= 30 and upper <= 0.2),
            "calibration_diagnostics": diagnostics(predictions, artifact["mapping"]),
            "production_qualified": False,
        }
        reports[name] = report
        all_rows[name] = {"raw": unqualified, "calibrated": rows}
        records.extend(
            {
                k: r.get(k)
                for k in [
                    "id",
                    "group_id",
                    "case_id",
                    "expected",
                    "candidate",
                    "decision",
                    "confidence",
                    "calibrated_correctness",
                    "raw_score",
                    "truncated",
                ]
            }
            | {"variant": name}
            for r in rows
        )
    selected = gate["model_selected"]
    comparisons = {}
    keys = sorted(expected)
    for name, data in all_rows.items():
        if name == selected:
            continue
        comparisons[name] = {}
        for policy in ["raw", "calibrated"]:
            b = {r["id"]: r for r in data[policy]}
            c = {r["id"]: r for r in all_rows[selected][policy]}
            bc = np.array([b[id]["decision"] == b[id]["expected"] for id in keys])
            cc = np.array([c[id]["decision"] == c[id]["expected"] for id in keys])
            bd = np.array(
                [
                    b[id]["decision"] == b[id]["expected"] and b[id]["decision"] in ["YES", "NO"]
                    for id in keys
                ]
            )
            cd = np.array(
                [
                    c[id]["decision"] == c[id]["expected"] and c[id]["decision"] in ["YES", "NO"]
                    for id in keys
                ]
            )
            rng = np.random.default_rng(1922)
            indices = rng.integers(0, len(keys), size=(5000, len(keys)))
            delta = cc.astype(int) - bc.astype(int)
            defdelta = cd.astype(int) - bd.astype(int)
            win = int(np.sum(cc & ~bc))
            loss = int(np.sum(bc & ~cc))
            p = float(binomtest(win, win + loss, 0.5).pvalue) if win + loss else 1.0
            comparisons[name][policy] = {
                "accuracy_delta": float(delta.mean()),
                "accuracy_delta_95_paired_interval": np.quantile(
                    delta[indices].mean(axis=1), [0.025, 0.975]
                ).tolist(),
                "correct_definitive_delta": float(defdelta.mean()),
                "correct_definitive_delta_95_paired_interval": np.quantile(
                    defdelta[indices].mean(axis=1), [0.025, 0.975]
                ).tolist(),
                "mcnemar_exact_p": p,
                "selected_only_correct": win,
                "baseline_only_correct": loss,
            }
    # A predeclared family of selected-vs-four-baseline raw accuracy comparisons; other intervals descriptive.
    ordered = sorted(comparisons, key=lambda k: comparisons[k]["raw"]["mcnemar_exact_p"])
    previous = 0.0
    for rank, name in enumerate(ordered):
        previous = max(
            previous, min(1.0, (len(ordered) - rank) * comparisons[name]["raw"]["mcnemar_exact_p"])
        )
        comparisons[name]["raw"]["holm_adjusted_p"] = previous
    output = {
        "selected_configuration": selected,
        "systems": reports,
        "paired_comparisons": comparisons,
        "scope": "Independent component-uniform conditional revision queries. Not current-world truth, live retrieval, all-case correctness or a Gemini workflow.",
        "gemini_comparison": None,
        "all_in_cost": None,
        "latency_comparison": None,
        "locked_model_selection_changed": False,
    }
    target.write_text(json.dumps(output, indent=2) + "\n")
    (safe / "locked-records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    print(
        json.dumps(
            {
                n: {
                    "raw_accuracy": v["raw"]["accuracy_all_decisions"],
                    "coverage": v["calibrated080"]["definitive_coverage"],
                    "correct_definitive": v["calibrated080"]["correct_definitive_fraction"],
                    "selected_errors": v["selected_errors"],
                    "selected_answers": v["selected_answers"],
                    "risk_upper": v["selective_risk_family_adjusted_upper"],
                }
                for n, v in reports.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
