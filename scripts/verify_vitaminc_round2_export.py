"""Attest local trained ONNX weights only after dynamic-shape parity checks."""

import hashlib
import json
from pathlib import Path

import numpy as np

from houndoracle.experiments.nli import OnnxNLI


def main():
    root = Path("/workspace/houndoracle-models/minilm-vitc-fever-1921")
    training = json.loads(
        Path("research/experiments/campaign-08/round-2/semantic-training.json").read_text()
    )
    if not training["completed"]:
        raise ValueError("Partial training is not the predeclared candidate")
    manifest = {
        "model_id": "houndoracle/minilm-vitc-fever-1921",
        "revision": training["checkpoint_sha256"],
        "declared_license": "Research checkpoint; base and dataset rights require separate redistribution review",
        "max_length": 256,
        "files": {},
    }
    for filename in ["config.json", "tokenizer.json", "onnx/model.onnx"]:
        path = root / filename
        manifest["files"][filename] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model = OnnxNLI(root, variant="fp32", threads=2, reviewed_manifest=manifest)
    cases = json.loads(
        Path("/workspace/houndoracle-data/campaign-08-round2/parity-cases.json").read_text()
    )
    checks = []
    for case in cases:
        values = {k: np.array(v, dtype=np.int64) for k, v in case["inputs"].items()}
        actual = np.array(
            model.session.run(None, {i.name: values[i.name] for i in model.session.get_inputs()})[0]
        )
        expected = np.array(case["torch_logits"])
        checks.append(
            {
                "batch": len(actual),
                "sequence_length": len(values["input_ids"][0]),
                "max_absolute_error": float(np.max(np.abs(actual - expected))),
                "argmax_matches": bool(np.all(actual.argmax(axis=1) == expected.argmax(axis=1))),
            }
        )
    passed = all(c["max_absolute_error"] < 1e-4 and c["argmax_matches"] for c in checks)
    result = {"passed": passed, "tolerance": 1e-4, "cases": checks, "model_config": model.config}
    Path("research/experiments/campaign-08/round-2/semantic-export-parity.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    if not passed:
        raise ValueError("Export parity failed; do not benchmark a different numerical policy")
    Path("research/experiments/campaign-08/round-2/trained-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
