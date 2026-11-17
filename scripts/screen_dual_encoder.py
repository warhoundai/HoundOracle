"""Fixed complementary-encoder screen; never opens fresh risk or locked test."""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from houndoracle.experiments.fit_calibration import fit, predict
from houndoracle.experiments.head import infer_head
from houndoracle.experiments.metrics import summary

PUBLIC = Path("research/experiments/campaign-27")
PRIVATE = Path("/workspace/houndoracle-data/campaign-27")


def load(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def pair(a, b):
    if len(a) != len(b):
        raise ValueError("Mismatched paired data")
    out = []
    for x, y in zip(a, b, strict=True):
        if any(x[k] != y[k] for k in ["id", "group_id", "expected"]):
            raise ValueError("Paired labels/groups mismatch")
        out.append((x, y))
    return out


def vector(a, b):
    x = a["logits"] or [0, 0, 1]
    y = b["logits"] or [0, 0, 1]
    return (
        x
        + [
            np.log1p(a.get("token_count") or a.get("model_input_tokens") or 0),
            np.log1p(a["evidence_group_count"]),
        ]
        + y
        + [abs(u - v) for u, v in zip(x, y)]
        + [float(a["candidate"] == b["candidate"])]
    )


def provisional(rows, mapping):
    return [
        r
        | {
            "decision": r["candidate"]
            if r["candidate"] in ["YES", "NO"]
            and not r.get("truncated")
            and predict(mapping, r["raw_score"]) >= 0.8
            else "MAYBE",
            "latency_seconds": None,
        }
        for r in rows
    ]


def main():
    if (PUBLIC / "screen.json").exists():
        raise FileExistsError("Preserve completed screen")
    training = pair(
        load("/workspace/houndoracle-data/campaign-08-round2/scores/continued/development.jsonl"),
        load("/workspace/houndoracle-data/campaign-19/scores/candidate/head-training.jsonl"),
    )
    validation = pair(
        load("/workspace/houndoracle-data/campaign-09/scores/continued/validation.jsonl"),
        load("/workspace/houndoracle-data/campaign-19/scores/candidate/validation.jsonl"),
    )
    calibration = pair(
        load("/workspace/houndoracle-data/campaign-09/scores/continued/calibration-fit.jsonl"),
        load("/workspace/houndoracle-data/campaign-19/scores/candidate/calibration-fit.jsonl"),
    )
    groups = [{a["group_id"] for a, b in rows} for rows in [training, validation, calibration]]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Partition leakage")
    scaler = StandardScaler().fit([vector(a, b) for a, b in training])
    model = LogisticRegression(C=1, max_iter=1000, random_state=1961).fit(
        scaler.transform([vector(a, b) for a, b in training]),
        [{"NO": 0, "YES": 1, "MAYBE": 2}[a["expected"]] for a, b in training],
    )
    head = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": model.coef_.tolist(),
        "biases": model.intercept_.tolist(),
        "classes": model.classes_.tolist(),
    }

    def score(rows):
        return [
            infer_head(
                a | {"truncated": a.get("truncated") or b.get("truncated")},
                head,
                evidence_group_count=a["evidence_group_count"],
                extra_features=vector(a, b)[5:],
            )
            for a, b in rows
        ]

    candidate = score(validation)
    cal = score(calibration)
    mapping = fit("logistic", [r for r in cal if not r.get("truncated")])
    basehead = json.loads(
        Path("research/experiments/campaign-09/continued-mixture-head.json").read_text()
    )
    baseline = [
        infer_head(a, basehead, evidence_group_count=a["evidence_group_count"])
        for a, b in validation
    ]
    basemap = json.loads(
        Path("research/experiments/campaign-09/continued-mixture-calibration.json").read_text()
    )["mapping"]
    raw = lambda rows: summary(
        [r | {"decision": r["candidate"], "latency_seconds": None} for r in rows]
    )
    a, b = raw(baseline), raw(candidate)
    ap, bp = provisional(baseline, basemap), provisional(candidate, mapping)
    ac, bc = summary(ap), summary(bp)
    delta = b["accuracy_all_decisions"] - a["accuracy_all_decisions"]
    gain = bc["correct_definitive_fraction"] - ac["correct_definitive_fraction"]
    passed = (
        (delta >= 0.02 or gain >= 0.02)
        and delta >= -0.01
        and bc["definitive_coverage"] >= ac["definitive_coverage"] - 0.02
    )
    report = {
        "baseline": a,
        "candidate": b,
        "provisional_baseline": ac,
        "provisional_candidate": bc,
        "raw_accuracy_delta": delta,
        "correct_definitive_delta": gain,
        "screen_passed": bool(passed),
        "decision": "qualify-fresh-risk" if passed else "reject-before-fresh-risk",
        "fresh_risk_opened": False,
        "locked_opened": False,
        "precision99_qualified": False,
        "scope": "Reused development screen, cached encoder timings not candidate latency; calibration provisional, citations unaudited; extra encoder inference adds compute.",
    }
    for name, data in [("head.json", head), ("mapping.json", mapping), ("screen.json", report)]:
        (PUBLIC / name).write_text(json.dumps(data, indent=2) + "\n")
    (PRIVATE / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in candidate))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
