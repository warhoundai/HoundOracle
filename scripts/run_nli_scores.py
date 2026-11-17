"""Generate predeclared model traces without opening the locked evaluation split."""

import argparse
import json
import time
from pathlib import Path

from houndoracle.experiments.nli import OnnxNLI


def run(directory, model_directory, output, variant):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = OnnxNLI(model_directory, variant=variant)
    (output / ("model-" + variant + ".json")).write_text(json.dumps(model.config, indent=2) + "\n")
    for split in ["development", "validation", "calibration"]:
        path = output / (split + "-" + variant + ".jsonl")
        rows = [
            json.loads(s) for s in (Path(directory) / (split + ".jsonl")).read_text().splitlines()
        ]
        if path.exists():
            previous = [json.loads(s) for s in path.read_text().splitlines()]
            if [r["id"] for r in previous] != [r["id"] for r in rows] or any(
                r["engine_config"] != model.config for r in previous
            ):
                raise ValueError(
                    "Incomplete, stale or mismatched existing trace; use a new output directory"
                )
            print("Verified complete trace, not replaying:", path, flush=True)
            continue
        with path.open("w") as handle:
            for i, row in enumerate(rows):
                start = time.perf_counter()
                pred = model.predict(row["question"], row["evidence"])
                item = {k: row[k] for k in ["id", "group_id", "expected", "category"]}
                item.update(
                    pred,
                    latency_seconds=time.perf_counter() - start,
                    engine="nli-" + variant,
                    source_urls=[e["url"] for e in row["evidence"]],
                    evidence_group_count=len({e["page"] for e in row["evidence"]}),
                    input_question=row["question"],
                    search_api_calls=0,
                    model_input_tokens=pred.get("token_count"),
                    model_output_tokens=0,
                    reasoning_tokens=0,
                    model_cost_usd=0,
                    search_cost_usd=0,
                    compute_cost_usd=None,
                    storage_cost_usd=None,
                    all_in_cost_usd=None,
                    cache_hit=False,
                    evidence_sufficiency_independently_labeled=row["expected"] != "MAYBE",
                    citation_supported=None,
                    citation_audit_method="not independently assessed",
                    label_agreement_on_supplied_evidence=(
                        pred["candidate"] == row["expected"] and pred["candidate"] in ("YES", "NO")
                    ),
                    retrieval_recall=None,
                    retrieval_mode="fixed supplied historical evidence",
                    failure=None,
                    engine_config=model.config,
                )
                handle.write(json.dumps(item) + "\n")
                handle.flush()
                if (i + 1) % 100 == 0:
                    print(variant, split, i + 1, "/", len(rows), flush=True)
        print("Completed", variant, split, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=["fp32", "int8"], required=True)
    args = p.parse_args()
    run(args.data, args.model, args.output, args.variant)
