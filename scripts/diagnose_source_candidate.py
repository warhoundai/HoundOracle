"""Retired distribution-shift challenge; explicitly not a fresh independent evaluation."""

import argparse
import json
import time
from pathlib import Path

from scipy.stats import beta

from houndoracle.experiments.head import infer_head
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.pipeline import apply
from houndoracle.experiments.source_context import SourceContextNLI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "variant",
        choices=[
            "source-original",
            "source-finetuned",
            "source-original-v2",
            "source-finetuned-v2",
        ],
    )
    variant = parser.parse_args().variant
    public = Path("benchmarks/results/campaign-07")
    target = public / f"{variant}-retired-challenge.json"
    if target.exists():
        raise FileExistsError("Already challenged; preserve outcome")
    head = json.loads(Path(f"research/experiments/campaign-07/{variant}-head.json").read_text())
    artifact = json.loads(
        Path(f"research/experiments/campaign-07/calibration-{variant}.json").read_text()
    )
    oldhead = json.loads(Path("research/experiments/results/trained-aggregators.json").read_text())[
        "logistic"
    ]
    oldartifact = json.loads(
        Path("research/experiments/campaign-04/calibration-logistic-threads2.json").read_text()
    )
    model = OnnxNLI("/workspace/houndoracle-models/minilm", variant="fp32", threads=2)
    fine = variant.startswith("source-finetuned")
    manifest = (
        json.loads(Path("research/model-comparisons/finetuned-minilm-provenance.json").read_text())
        if fine
        else None
    )
    candidate = SourceContextNLI(
        OnnxNLI(
            "/workspace/houndoracle-models/" + ("minilm-fever-1912" if fine else "minilm"),
            variant="fp32",
            threads=2,
            reviewed_manifest=manifest,
        )
    )
    results = {}
    records = []
    for panel, path in [
        ("balanced", "/workspace/houndoracle-data/campaign-06-balanced/locked.jsonl"),
        ("strict", "/workspace/houndoracle-data/campaign-06/locked.jsonl"),
    ]:
        rows = [json.loads(s) for s in Path(path).read_text().splitlines()]
        stats = {
            name: {"selected": 0, "errors": 0, "n": len(rows), "correct_all": 0, "seconds": 0.0}
            for name in ["baseline", "grounding"]
        }
        for row in rows:
            base = model.predict(row["question"], row["evidence"])
            groups = len({e.get("page") for e in row["evidence"]})
            for name, checkpoint, cal, extra in [
                ("baseline", oldhead, oldartifact, []),
                ("grounding", head, artifact, []),
            ]:
                start = time.perf_counter()
                result = apply(
                    infer_head(
                        candidate.predict(row["question"], row["evidence"])
                        if name == "grounding"
                        else base,
                        checkpoint,
                        evidence_group_count=groups,
                        extra_features=extra,
                    ),
                    cal,
                )
                accepted = (
                    result["decision"] in ["YES", "NO"] and (result.get("confidence") or 0) >= 0.85
                )
                decision = result["decision"] if accepted else "MAYBE"
                stats[name]["selected"] += accepted
                stats[name]["errors"] += accepted and decision != row["expected"]
                stats[name]["correct_all"] += decision == row["expected"]
                stats[name]["seconds"] += time.perf_counter() - start
                records.append(
                    {
                        "id": row["id"],
                        "panel": panel,
                        "arm": name,
                        "expected": row["expected"],
                        "decision": decision,
                        "accepted": accepted,
                    }
                )
        for s in stats.values():
            k, n = s["errors"], s["selected"]
            s["selective_risk"] = k / n if n else None
            s["one_sided_error_upper"] = float(beta.ppf(0.95, k + 1, n - k)) if n and k < n else 1.0
            s["correct_definitive_fraction"] = (n - k) / s["n"]
        results[panel] = stats
        print(json.dumps({panel: stats}), flush=True)
    results["fresh"] = False
    results["interpretation"] = (
        "Retired challenge only; no post-hoc threshold tuning or production qualification. Head-only timing excludes shared encoder; no end-to-end speed claim."
    )
    target.write_text(json.dumps(results, indent=2) + "\n")
    target.with_name(f"{variant}-retired-records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )


if __name__ == "__main__":
    main()
