"""Fixed .85 decision screens with paired quality uncertainty, no latency claims."""

import json
import random
from pathlib import Path

import numpy as np

from houndoracle.experiments.metrics import summary
from houndoracle.experiments.pipeline import apply


def load(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def at85(rows):
    return [
        r
        | {
            "decision": r["decision"]
            if r["decision"] in ["YES", "NO"] and (r.get("confidence") or 0) >= 0.85
            else "MAYBE"
        }
        for r in rows
    ]


def main():
    output = Path("research/experiments/campaign-07")
    oldcal = json.loads(
        Path("research/experiments/campaign-04/calibration-logistic-threads2.json").read_text()
    )
    base = at85(
        [
            apply(r, oldcal)
            for r in load("/workspace/houndoracle-data/scores/validation-logistic.jsonl")
        ]
    )
    summaries = {}
    for name in ["source-original-v2", "source-finetuned-v2", "grounding-head"]:
        path = (
            Path("/workspace/houndoracle-data/campaign-07")
            / name
            / f"validation-calibrated-{name}.jsonl"
        )
        if not path.exists():
            continue
        candidate = at85(load(path))
        b = {r["id"]: r for r in base}
        c = {r["id"]: r for r in candidate}
        assert b.keys() == c.keys()
        groups = {}
        for id, row in b.items():
            assert row["group_id"] == c[id]["group_id"]
            groups.setdefault(row["group_id"], []).append(id)
        rng = random.Random(1918)
        gkeys = sorted(groups)
        draws = []
        for _ in range(2000):
            ids = [id for group in rng.choices(gkeys, k=len(gkeys)) for id in groups[group]]
            draws.append(
                sum(
                    (c[id]["decision"] == c[id]["expected"] and c[id]["decision"] in ["YES", "NO"])
                    - (
                        b[id]["decision"] == b[id]["expected"]
                        and b[id]["decision"] in ["YES", "NO"]
                    )
                    for id in ids
                )
                / len(ids)
            )
        bs, cs = summary(base), summary(candidate)
        for s in [bs, cs]:
            s["latency_seconds"] = None
            s["citation_supported_definitive_fraction"] = None
        gain = cs["correct_definitive_fraction"] - bs["correct_definitive_fraction"]
        passed = (
            gain >= 0.02
            and cs["accuracy_all_decisions"] >= bs["accuracy_all_decisions"]
            and cs["selective_risk"] is not None
            and cs["selective_risk"] <= bs["selective_risk"]
        )
        summaries[name] = {
            "baseline": bs,
            "candidate": cs,
            "correct_definitive_gain": gain,
            "paired_group_bootstrap_95_gain_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
            "screen_passed": passed,
            "independent": False,
            "production_qualified": False,
            "note": "Repeated old validation screen; interval is exploratory, not multiplicity-adjusted independent proof. Latency and citation scores withheld: inherited traces do not establish those comparisons.",
        }
    (output / "corrected-screen-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {a: b for a, b in v.items() if a not in ["baseline", "candidate"]}
                for k, v in summaries.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
