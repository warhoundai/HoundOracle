"""Score stable reference on the gated sealed cohort without changing its policy."""

import hashlib
import json
from pathlib import Path

from houndoracle.experiments.head import infer_head
from houndoracle.experiments.nli import OnnxNLI


def main():
    public = Path("research/experiments/campaign-29")
    risk = json.loads((public / "risk-assessment.json").read_text())
    if not risk["conditional80_research_gate_passed"]:
        raise PermissionError("No passing independent risk gate")
    source = Path("/workspace/houndoracle-data/campaign-19/locked.jsonl")
    if (
        hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
        != "56de0d87f6e76493748b7fbc2280dc508a94142a5b9083613a37592b4cf4a1bf"
    ):
        raise ValueError("Sealed cohort changed")
    out = Path("/workspace/houndoracle-data/campaign-29/baseline-locked-scores.jsonl")
    if out.exists():
        raise FileExistsError("Preserve completed reference observations")
    manifest = json.loads(
        Path("research/experiments/campaign-08/round-2/trained-manifest.json").read_text()
    )
    model = OnnxNLI(
        "/workspace/houndoracle-models/minilm-vitc-fever-1921",
        variant="fp32",
        threads=1,
        max_length=256,
        reviewed_manifest=manifest,
    )
    head = json.loads(
        Path("research/experiments/campaign-09/continued-mixture-head.json").read_text()
    )
    with out.open("x") as handle:
        for i, line in enumerate(source.read_text().splitlines()):
            row = json.loads(line)
            count = len({e["page"] for e in row["evidence"]})
            raw = model.predict(row["question"], row["evidence"])
            pred = infer_head(raw, head, evidence_group_count=count)
            result = (
                {k: row[k] for k in ["id", "group_id", "expected"]}
                | pred
                | {"evidence_group_count": count, "citation_supported": None}
            )
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            if (i + 1) % 100 == 0:
                print(json.dumps({"reference_scored": i + 1}), flush=True)


if __name__ == "__main__":
    main()
