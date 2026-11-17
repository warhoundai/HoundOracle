"""Record raw matched-evidence predictions; locked data requires explicit qualified gate."""

import argparse
import hashlib
import json
from pathlib import Path

from houndoracle.experiments.nli import OnnxNLI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", choices=["original", "contrastive", "fever-finetuned", "continued"]
    )
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--round2", action="store_true")
    args = parser.parse_args()
    if args.model == "continued" and not args.round2:
        parser.error("continued requires --round2")
    public = Path("research/experiments/campaign-08")
    if args.round2:
        public = public / "round-2"
    if args.locked:
        gate = json.loads((public / "locked-authorization.json").read_text())
        if gate.get("qualified") is not True:
            raise PermissionError("Locked assessment not yet justified")
        for name, expected in gate["file_sha256"].items():
            if hashlib.sha256((public / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen policy or dataset changed after qualification")
    manifest_path = {
        "contrastive": Path("research/experiments/campaign-08/trained-manifest.json"),
        "continued": public / "trained-manifest.json",
        "fever-finetuned": Path("research/model-comparisons/finetuned-minilm-provenance.json"),
    }.get(args.model)
    manifest = json.loads(manifest_path.read_text()) if manifest_path else None
    directory = {
        "original": "minilm",
        "contrastive": "minilm-vitc-fever-1920",
        "continued": "minilm-vitc-fever-1921",
        "fever-finetuned": "minilm-fever-1912",
    }[args.model]
    model = OnnxNLI(
        "/workspace/houndoracle-models/" + directory,
        variant="fp32",
        threads=1,
        reviewed_manifest=manifest,
        max_length=256,
    )
    root = Path(
        "/workspace/houndoracle-data/campaign-08-round2"
        if args.round2
        else "/workspace/houndoracle-data/campaign-08"
    )
    target = root / "scores" / args.model
    target.mkdir(parents=True, exist_ok=True)
    config_path = target / "model.json"
    if config_path.exists() and json.loads(config_path.read_text()) != model.config:
        raise ValueError("Existing score directory belongs to a different encoder configuration")
    if not config_path.exists():
        config_path.write_text(json.dumps(model.config, indent=2) + "\n")
    for split in (
        ["locked"]
        if args.locked
        else (
            ["validation", "calibration-fit", "calibration-risk"]
            if args.model == "fever-finetuned"
            else ["development", "validation", "calibration-fit", "calibration-risk"]
        )
    ):
        path = target / (split + ".jsonl")
        if path.exists():
            raise FileExistsError("Do not silently rescore or tune frozen runs")
        source = root / (split + ".jsonl")
        extended = public / "dataset-manifest-extended.json"
        if args.round2 and split == "calibration-risk" and extended.exists():
            source = Path(json.loads(extended.read_text())["risk_source"])
        seal = json.loads(
            (
                public
                / (
                    "dataset-manifest-extended.json"
                    if extended.exists()
                    else "dataset-manifest.json"
                )
            ).read_text()
        )
        expected_input_sha = seal["panels"][split]["sha256"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected_input_sha:
            raise ValueError("Dataset split differs from sealed input snapshot")
        if (
            args.round2
            and args.model == "contrastive"
            and split in ["validation", "calibration-fit"]
        ):
            parent = Path("/workspace/houndoracle-data/campaign-08")
            if (
                hashlib.sha256(source.read_bytes()).hexdigest()
                != hashlib.sha256((parent / (split + ".jsonl")).read_bytes()).hexdigest()
            ):
                raise ValueError("Input changed; cannot reuse parent encoder outputs")
            if json.loads((parent / "scores/contrastive/model.json").read_text()) != model.config:
                raise ValueError("Parent score encoder configuration differs")
            path.write_bytes((parent / "scores/contrastive" / (split + ".jsonl")).read_bytes())
            continue
        with path.open("w") as out:
            for i, line in enumerate(source.open()):
                row = json.loads(line)
                pred = model.predict(row["question"], row["evidence"])
                item = (
                    {k: row[k] for k in ["id", "group_id", "expected", "category"]}
                    | {"case_id": row.get("case_id", row["group_id"])}
                    | pred
                    | {
                        "latency_seconds": pred["model_seconds"],
                        "evidence_group_count": len({e["page"] for e in row["evidence"]}),
                        "model_input_tokens": pred.get("token_count"),
                        "all_in_cost_usd": None,
                        "citation_supported": None,
                        "timing_context": "Concurrent local research workloads; no paired latency inference from this trace",
                    }
                )
                out.write(json.dumps(item) + "\n")
                if (i + 1) % 400 == 0:
                    print(
                        json.dumps({"model": args.model, "split": split, "rows": i + 1}), flush=True
                    )
        print(
            json.dumps(
                {"completed": split, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
