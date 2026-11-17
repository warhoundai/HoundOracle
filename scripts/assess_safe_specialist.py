"""One-shot successor assessment; no provider calls or inferred Gemini outcomes."""

import json
from pathlib import Path

import numpy as np

from houndoracle.experiments.abstention import AbstentionGate
from houndoracle.experiments.head import HeadNLI
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.pipeline import ResearchPipeline
from houndoracle.experiments.router import OracleFirstRouter


class UnexecutedFallback:
    def answer(self, *args):
        return {"answer": {"decision": "MAYBE"}, "token_cost_at_list_price_usd": None}


def main():
    root = Path("research/experiments/campaign-06")
    target = Path("benchmarks/results/campaign-06/safe-specialist-summary.json")
    if target.exists():
        raise FileExistsError("Successor already evaluated; no retuning")

    def load(p):
        return json.loads(Path(p).read_text())

    old = ResearchPipeline(
        HeadNLI(
            OnnxNLI("/workspace/houndoracle-models/minilm", variant="fp32", threads=2),
            load("research/experiments/results/trained-aggregators.json")["logistic"],
            "logistic",
        ),
        load("research/experiments/campaign-04/calibration-logistic-threads2.json"),
    )
    fine = ResearchPipeline(
        HeadNLI(
            OnnxNLI(
                "/workspace/houndoracle-models/minilm-fever-1912",
                variant="fp32",
                threads=2,
                reviewed_manifest=load(
                    "research/model-comparisons/finetuned-minilm-provenance.json"
                ),
            ),
            load(root / "finetuned-head.json"),
            "finetuned-logistic",
        ),
        load(root / "calibration-finetuned.json"),
    )
    policy = load(root / "assessed-policy.json")
    print(list(policy), flush=True)
    gate = AbstentionGate(policy["insufficiency_gate"])
    router = OracleFirstRouter(
        old,
        UnexecutedFallback(),
        minimum_confidence=0.85,
        complete_evidence=True,
        insufficiency_specialist=fine,
        insufficiency_gate=gate,
    )
    records, summaries = [], {}
    for panel in ["balanced-risk", "strict-risk", "balanced-locked", "strict-locked"]:
        rows = [
            json.loads(s)
            for s in Path("/workspace/houndoracle-data/campaign-06-successor", panel + ".jsonl")
            .read_text()
            .splitlines()
        ]
        panel_records = []
        for i, row in enumerate(rows):
            result = router.answer(row["question"], row["evidence"], proposition_validated=True)
            base = result["oracle"]
            accepted = bool(
                base and base["decision"] in ["YES", "NO"] and (base["confidence"] or 0) >= 0.85
            )
            if accepted:
                assert result["source"] == "oracle" and result["decision"] == base["decision"]
            item = {
                "id": row["id"],
                "group_id": row["group_id"],
                "panel": panel,
                "expected": row["expected"],
                "source": result["source"],
                "decision": result["decision"] if result["source"] != "gemini_fallback" else None,
                "baseline_definitive": accepted,
                "latency_seconds": result["latency_seconds"],
                "additional_abstention": result["source"] == "calibrated_insufficiency_specialist",
            }
            panel_records.append(item)
            if (i + 1) % 100 == 0:
                print(json.dumps({"panel": panel, "rows": i + 1}), flush=True)
        n = len(rows)
        skipped = [r for r in panel_records if r["additional_abstention"]]
        summaries[panel] = {
            "n": n,
            "baseline_definitive": sum(r["baseline_definitive"] for r in panel_records),
            "new_definitive": 0,
            "additional_calls_avoided": len(skipped),
            "additional_calls_avoided_fraction": len(skipped) / n,
            "false_abstentions": sum(r["expected"] != "MAYBE" for r in skipped),
            "local_p50_seconds": float(
                np.quantile([r["latency_seconds"] for r in panel_records], 0.5)
            ),
            "local_p95_seconds": float(
                np.quantile([r["latency_seconds"] for r in panel_records], 0.95)
            ),
            "gemini_accuracy": None,
            "all_in_cost": None,
            "note": "Fallback not executed. No end-to-end Gemini advantage established. Definitive baseline preserved, not independently requalified.",
        }
        records += panel_records
    target.write_text(json.dumps(summaries, indent=2) + "\n")
    target.with_name("safe-specialist-records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
