"""Fit/choose once on old development material, freeze before opening fresh risk/locked outcomes."""

import json
from pathlib import Path

from houndoracle.experiments.abstention import policy_digest, would_abstain
from houndoracle.experiments.fit_calibration import fit


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    output = Path("research/experiments/campaign-06/abstention-frozen.json")
    if output.exists():
        raise FileExistsError("Gate already frozen; do not retune against fresh risk/test")
    old = json.loads(Path("research/experiments/results/calibration-logistic.json").read_text())
    runtime = json.loads(
        Path("research/experiments/campaign-04/calibration-logistic-threads2.json").read_text()
    )
    rows = load(Path("/workspace/houndoracle-data/scores/calibration-logistic.jsonl"))
    fitting = [
        {**r, "candidate": "YES", "expected": "YES" if r["expected"] == "MAYBE" else "NO"}
        for r in rows
        if r["group_id"] in set(old["fit_group_ids"])
        and r["candidate"] == "MAYBE"
        and not r.get("truncated")
    ]
    mapping = fit("logistic", fitting)
    valid = load(Path("/workspace/houndoracle-data/scores/validation-logistic.jsonl"))
    candidates = []
    for threshold in [0.90, 0.95, 0.975, 0.99]:
        policy = {"mapping": mapping, "threshold": threshold}
        selected = [r for r in valid if would_abstain(r, policy)]
        errors = sum(r["expected"] != "MAYBE" for r in selected)
        candidates.append(
            {
                "threshold": threshold,
                "selected": len(selected),
                "false_abstentions": errors,
                "lost_definitive_fraction_upper_proxy": errors / len(valid),
                "avoided_fraction": len(selected) / len(valid),
            }
        )
    acceptable = [c for c in candidates if c["lost_definitive_fraction_upper_proxy"] <= 0.01]
    selected = (
        max(acceptable, key=lambda c: (c["selected"], c["threshold"])) if acceptable else None
    )
    artifact = {
        "schema_version": 1,
        "mapping": mapping,
        "threshold": selected["threshold"] if selected else 0.99,
        "model_config": runtime["model_config"],
        "verification_policy_digest": policy_digest(runtime),
        "population": "source-disjoint FEVER residual components, historical supplied evidence only",
        "risk_certificate": {"qualified": False, "reason": "Fresh risk set not evaluated"},
        "production_qualified": False,
    }
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    Path("research/experiments/campaign-06/abstention-selection.json").write_text(
        json.dumps(
            {
                "fit_rows": len(fitting),
                "fit_groups": len({r["group_id"] for r in fitting}),
                "candidates": candidates,
                "selected": selected,
                "note": "Fresh risk and locked data were never opened for threshold selection. Error rate is an upper proxy for lost definitive coverage, not a new truth probability.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps({"mapping": mapping, "candidates": candidates, "selected": selected}, indent=2)
    )


if __name__ == "__main__":
    main()
