"""Paired parity and provisional timing screen; no new calibration or test decisions."""

import json
import random
import time
from pathlib import Path

import numpy as np

from houndoracle.experiments.head import HeadNLI
from houndoracle.experiments.nli import OnnxNLI


def main():
    root = Path("research/experiments/campaign-13")
    safe = Path("benchmarks/results/campaign-13")
    if (safe / "screen.json").exists():
        raise FileExistsError("Graph screen already measured")
    head = json.loads(
        Path("research/experiments/campaign-09/continued-mixture-head.json").read_text()
    )
    models = {}
    for name, directory, manifest in [
        (
            "source",
            "minilm-vitc-fever-1921",
            "research/experiments/campaign-08/round-2/trained-manifest.json",
        ),
        ("fused", "minilm-vitc-fever-1921-fused", str(root / "fused-manifest.json")),
    ]:
        models[name] = HeadNLI(
            OnnxNLI(
                "/workspace/houndoracle-models/" + directory,
                variant="fp32",
                threads=1,
                max_length=256,
                reviewed_manifest=json.loads(Path(manifest).read_text()),
            ),
            head,
            "continued-mixture",
        )
    rows = [
        json.loads(line)
        for line in Path("/workspace/houndoracle-data/campaign-09/validation.jsonl")
        .read_text()
        .splitlines()
    ]
    for row in rows[:10]:
        for model in models.values():
            model.predict(row["question"], row["evidence"])
    records = []
    rng = random.Random(1926)
    max_error = 0.0
    same = True
    for repeat in range(3):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for i in order:
            row = rows[i]
            names = list(models)
            rng.shuffle(names)
            results = {}
            for name in names:
                start = time.perf_counter()
                result = models[name].predict(row["question"], row["evidence"])
                seconds = time.perf_counter() - start
                results[name] = result
                records.append(
                    {"id": row["id"], "repeat": repeat, "variant": name, "seconds": seconds}
                )
            if repeat == 0:
                max_error = max(
                    max_error,
                    float(
                        np.max(
                            np.abs(
                                np.array(results["source"]["encoder_logits"])
                                - np.array(results["fused"]["encoder_logits"])
                            )
                        )
                    ),
                )
                same = same and results["source"]["candidate"] == results["fused"]["candidate"]
    summaries = {
        name: {
            "p50_seconds": float(
                np.quantile([r["seconds"] for r in records if r["variant"] == name], 0.5)
            ),
            "p95_seconds": float(
                np.quantile([r["seconds"] for r in records if r["variant"] == name], 0.95)
            ),
        }
        for name in models
    }
    p50 = 1 - summaries["fused"]["p50_seconds"] / summaries["source"]["p50_seconds"]
    p95 = 1 - summaries["fused"]["p95_seconds"] / summaries["source"]["p95_seconds"]
    parity = bool(max_error <= 1e-4 and same)
    passed = bool(parity and p50 >= 0.15 and p95 >= 0.15)
    output = {
        "parity_passed": parity,
        "max_absolute_encoder_logit_error": max_error,
        "head_labels_identical": bool(same),
        "summaries": summaries,
        "p50_reduction": p50,
        "p95_reduction": p95,
        "screen_passed": passed,
        "decision": "qualify-further" if passed else "reject-no-material-speed-or-parity",
        "locked_scored": False,
        "scope": "Provisional paired raw encoder+head timing, warm models, one thread, three repeats; not an end-to-end agent/Gemini measurement.",
    }
    (safe / "screen.json").write_text(json.dumps(output, indent=2) + "\n")
    (safe / "screen-records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
