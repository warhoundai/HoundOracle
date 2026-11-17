import argparse
import json
import time
from pathlib import Path

from houndoracle.experiments.head import head_digest, infer_head

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--heads", required=True)
    p.add_argument("--variant", default="fp32", choices=["fp32", "int8"])
    args = p.parse_args()
    root = Path(args.scores)
    heads = json.loads(Path(args.heads).read_text())
    for name, checkpoint in heads.items():
        config = json.loads((root / ("model-" + args.variant + ".json")).read_text())
        config.update(aggregation_head=name, head_digest=head_digest(checkpoint))
        output_name = name if args.variant == "fp32" else name + "-int8"
        (root / ("model-" + output_name + ".json")).write_text(json.dumps(config, indent=2) + "\n")
        for split in ["development", "validation", "calibration"]:
            rows = [
                json.loads(s)
                for s in (root / (split + "-" + args.variant + ".jsonl")).read_text().splitlines()
            ]
            with (root / (split + "-" + output_name + ".jsonl")).open("w") as out:
                for row in rows:
                    start = time.perf_counter()
                    result = infer_head(
                        row, checkpoint, evidence_group_count=row["evidence_group_count"]
                    )
                    result["latency_seconds"] = row["latency_seconds"] + time.perf_counter() - start
                    result["engine_config"] = config
                    result["engine"] = name
                    result["label_agreement_on_supplied_evidence"] = result["candidate"] == row[
                        "expected"
                    ] and result["candidate"] in ("YES", "NO")
                    result["citation_supported"] = None
                    result["citation_audit_method"] = "not independently assessed"
                    out.write(json.dumps(result) + "\n")
