"""Counterfactual token accounting only; never reports replay as new cloud measurement."""

import json
from pathlib import Path


def main():
    reports = []
    records = []
    for suffix in ["", "b"]:
        raw = json.loads(
            Path(f"/workspace/houndoracle-data/cloud-03{suffix}-analysis.json").read_text()
        )
        payload_suffix = "-supplemental" if suffix else ""
        payload = json.loads(
            Path(
                f"/workspace/houndoracle-data/cloud-comparison-payload{payload_suffix}.json"
            ).read_text()
        )
        examples = {r["id"]: r for p in ["development", "confirmation"] for r in payload[p]}
        groups = {}
        for row in raw["records"]:
            empty = examples[row["id"]]["evidence"] == []
            # Human-labeled FEVER claim format establishes evaluability, not the expected verdict.
            decision = "MAYBE" if empty else row["decision"]
            item = {
                k: row[k]
                for k in [
                    "id",
                    "phase",
                    "arm",
                    "repeat",
                    "expected",
                    "decision",
                    "model_calls",
                    "token_cost_usd",
                ]
            }
            item.update(
                campaign=raw["campaign"],
                empty_gate=empty,
                counterfactual_decision=decision,
                counterfactual_calls=0 if empty else row["model_calls"],
                counterfactual_token_cost_usd=0 if empty else row["token_cost_usd"],
            )
            records.append(item)
            groups.setdefault(row["phase"] + ":" + row["arm"], []).append(item)
        for group, rows in groups.items():
            reports.append(
                {
                    "campaign": raw["campaign"],
                    "system": group,
                    "tasks": len(rows),
                    "changed_decisions": sum(
                        r["decision"] != r["counterfactual_decision"] for r in rows
                    ),
                    "original_model_calls": sum(r["model_calls"] for r in rows),
                    "counterfactual_model_calls": sum(r["counterfactual_calls"] for r in rows),
                    "original_token_cost_usd": sum(r["token_cost_usd"] for r in rows),
                    "counterfactual_token_cost_usd": sum(
                        r["counterfactual_token_cost_usd"] for r in rows
                    ),
                    "new_live_latency_measurement": None,
                }
            )
    out = Path("benchmarks/results/campaign-04")
    (out / "archived-replay.json").write_text(
        json.dumps(
            {
                "kind": "COUNTERFACTUAL_ONLY",
                "systems": reports,
                "fair_control": "Apply the identical complete-corpus empty-evidence gate to both standalone Gemini and Oracle-assisted arms. No benefit from deliberately wasting baseline calls.",
                "limitations": [
                    "No new Vertex calls, prices, end-to-end latency, invoice or independent locked evaluation.",
                    "An empty permitted corpus implies abstention only for an already evaluable proposition under evidence-only policy.",
                ],
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
