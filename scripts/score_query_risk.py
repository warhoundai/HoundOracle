"""Fixed independent-query semantic scores; no fitting and no implicit locked access."""

import argparse
import hashlib
import json
from pathlib import Path

from houndoracle.experiments.nli import OnnxNLI

MODELS = {
    "int8-mixed1": (
        "minilm-vitc-fever-1921-int8-mixed1",
        "research/experiments/campaign-12/int8-mixed1-manifest.json",
    ),
    "int8-mixed2": (
        "minilm-vitc-fever-1921-int8-mixed2",
        "research/experiments/campaign-12/int8-mixed2-manifest.json",
    ),
    "int8-mixed3": (
        "minilm-vitc-fever-1921-int8-mixed3",
        "research/experiments/campaign-12/int8-mixed3-manifest.json",
    ),
    "int8-dense-pc": (
        "minilm-vitc-fever-1921-int8-dense-pc",
        "research/experiments/campaign-11/int8-dense-pc-manifest.json",
    ),
    "int8-full-pc": (
        "minilm-vitc-fever-1921-int8-full-pc",
        "research/experiments/campaign-11/int8-full-pc-manifest.json",
    ),
    "fp32": (
        "minilm-vitc-fever-1921",
        "research/experiments/campaign-08/round-2/trained-manifest.json",
    ),
    "int8-dense": (
        "minilm-vitc-fever-1921-int8-dense",
        "research/experiments/campaign-10/int8-dense-manifest.json",
    ),
    "int8-full": (
        "minilm-vitc-fever-1921-int8-full",
        "research/experiments/campaign-10/int8-full-manifest.json",
    ),
    "original": ("minilm", None),
    "fever": ("minilm-fever-1912", "research/model-comparisons/finetuned-minilm-provenance.json"),
    "parent": ("minilm-vitc-fever-1920", "research/experiments/campaign-08/trained-manifest.json"),
    "continued": (
        "minilm-vitc-fever-1921",
        "research/experiments/campaign-08/round-2/trained-manifest.json",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--pairs", action="store_true")
    parser.add_argument("--quantization", action="store_true")
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()
    args.quantization = args.quantization or args.per_channel or args.mixed
    if (args.model == "fp32" or args.model.startswith("int8")) != args.quantization:
        parser.error("Quantization models require an explicit experiment flag")
    number = 12 if args.mixed else 11 if args.per_channel else 10 if args.quantization else 9
    public = Path(f"research/experiments/campaign-{number:02d}")
    root = Path(f"/workspace/houndoracle-data/campaign-{number:02d}")
    if args.pairs and not args.locked:
        parser.error("--pairs requires --locked")
    if args.locked:
        gate = json.loads((public / "locked-authorization.json").read_text())
        if gate.get("qualified") is not True:
            raise PermissionError("No qualified frozen policy")
        for name, sha in gate["file_sha256"].items():
            if hashlib.sha256((public / name).read_bytes()).hexdigest() != sha:
                raise ValueError("Frozen query policy changed")
    directory, manifest_path = MODELS[args.model]
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else None
    model = OnnxNLI(
        "/workspace/houndoracle-models/" + directory,
        variant="int8" if args.model.startswith("int8") else "fp32",
        threads=1,
        max_length=256,
        reviewed_manifest=manifest,
    )
    target = root / "scores" / args.model
    target.mkdir(parents=True, exist_ok=True)
    config = target / "model.json"
    if config.exists() and json.loads(config.read_text()) != model.config:
        raise ValueError("Encoder changed")
    if not config.exists():
        config.write_text(json.dumps(model.config, indent=2) + "\n")
    seal = json.loads((public / "dataset-manifest.json").read_text())
    for split in (
        ["locked-pairs"]
        if args.pairs
        else ["locked"]
        if args.locked
        else (
            ["validation", "calibration-risk"]
            if args.quantization
            else ["validation", "calibration-fit", "calibration-risk"]
        )
    ):
        path = target / (split + ".jsonl")
        if path.exists():
            raise FileExistsError("Preserve completed/partial runs; do not silently rescore")
        source = root / (split + ".jsonl")
        if hashlib.sha256(source.read_bytes()).hexdigest() != seal["panels"][split]["sha256"]:
            raise ValueError("Input snapshot changed")
        cache = {}
        if args.pairs:
            for line in (target / "locked.jsonl").read_text().splitlines():
                row = json.loads(line)
                cache[row["id"]] = row
        with path.open("w") as out:
            for i, line in enumerate(source.open()):
                row = json.loads(line)
                signature = hashlib.sha256(
                    json.dumps(
                        [row["question"], row["evidence"], model.config], sort_keys=True
                    ).encode()
                ).hexdigest()
                if row["id"] in cache:
                    prediction = cache[row["id"]]
                    if prediction["input_sha256"] != signature:
                        raise ValueError("Paired evidence is not exactly equivalent")
                    item = prediction | {"raw_cache_hit": True}
                else:
                    pred = model.predict(row["question"], row["evidence"])
                    item = (
                        {k: row[k] for k in ["id", "group_id", "case_id", "expected", "category"]}
                        | pred
                        | {
                            "input_sha256": signature,
                            "latency_seconds": pred["model_seconds"],
                            "evidence_group_count": len({e["page"] for e in row["evidence"]}),
                            "model_input_tokens": pred.get("token_count"),
                            "all_in_cost_usd": None,
                            "citation_supported": None,
                            "raw_cache_hit": False,
                            "timing_context": "Contended research collection; no paired speed claim",
                        }
                    )
                out.write(json.dumps(item) + "\n")
                if (i + 1) % 300 == 0:
                    print(json.dumps({"model": args.model, "split": split, "n": i + 1}), flush=True)
        print(
            json.dumps(
                {"completed": split, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
