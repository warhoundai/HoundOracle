"""Label-independent archived citation audit; postprocessor replay, not new inference."""

import json
from pathlib import Path

from houndoracle.experiments.citations import citations_match


def audit(analysis, payload):
    examples = {r["id"]: r for part in ["development", "confirmation"] for r in payload[part]}
    groups = {}
    for row in analysis["records"]:
        model = row.get("gemini")
        if not model:
            continue
        answer = model["answer"]
        definitive = answer["decision"] in ["YES", "NO"]
        valid = not definitive or citations_match(answer, examples[row["id"]]["evidence"])
        group = groups.setdefault(
            row["phase"] + ":" + row["arm"],
            {"model_responses": 0, "definitive": 0, "blocked_unverifiable_definitive": 0},
        )
        group["model_responses"] += 1
        group["definitive"] += definitive
        group["blocked_unverifiable_definitive"] += not valid
    return groups


def main():
    results = {}
    for suffix, payload_suffix in [("", ""), ("b", "-supplemental")]:
        analysis = json.loads(
            Path(f"/workspace/houndoracle-data/cloud-03{suffix}-analysis.json").read_text()
        )
        payload = json.loads(
            Path(
                f"/workspace/houndoracle-data/cloud-comparison-payload{payload_suffix}.json"
            ).read_text()
        )
        results["oracle-cloud-03" + suffix] = audit(analysis, payload)
    report = {
        "kind": "COUNTERFACTUAL_CITATION_POSTPROCESSOR_AUDIT",
        "results": results,
        "limitations": [
            "No new model generation, quality labeling or live latency measurement.",
            "URL/quote presence does not establish entailment or source independence.",
            "The submitted campaign05 is unchanged; audit both arms equally after collecting its frozen outputs.",
        ],
    }
    Path("benchmarks/results/campaign-04/citation-guard-audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
