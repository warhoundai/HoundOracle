"""Two fixed integer variants of the frozen selected encoder; no labels or training."""

import hashlib
import json
import shutil
import time
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    source = Path("/workspace/houndoracle-models/minilm-vitc-fever-1921")
    parent = json.loads(
        Path("research/experiments/campaign-08/round-2/trained-manifest.json").read_text()
    )
    if (
        hashlib.sha256((source / "onnx/model.onnx").read_bytes()).hexdigest()
        != parent["files"]["onnx/model.onnx"]["sha256"]
    ):
        raise ValueError("Selected source model changed")
    for name, ops in [("int8-dense", ["MatMul"]), ("int8-full", ["MatMul", "Gather"])]:
        target = source.with_name("minilm-vitc-fever-1921-" + name)
        if target.exists():
            raise FileExistsError("Quantization variant already exists")
        (target / "onnx").mkdir(parents=True)
        for file in ["config.json", "tokenizer.json"]:
            shutil.copyfile(source / file, target / file)
        start = time.perf_counter()
        quantize_dynamic(
            source / "onnx/model.onnx",
            target / "onnx/model_quint8_avx2.onnx",
            op_types_to_quantize=ops,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        model = target / "onnx/model_quint8_avx2.onnx"
        manifest = {
            "model_id": "houndoracle/minilm-vitc-fever-1921-" + name,
            "revision": hashlib.sha256(model.read_bytes()).hexdigest(),
            "max_length": 256,
            "declared_license": "Local research checkpoint; base/data redistribution compatibility review required",
            "quantization": {
                "source_onnx_sha256": parent["files"]["onnx/model.onnx"]["sha256"],
                "runtime": "1.32.0",
                "ops": ops,
                "weight_type": "QUInt8",
                "per_channel": False,
                "reduce_range": False,
                "seconds": time.perf_counter() - start,
            },
            "files": {},
        }
        for file in ["config.json", "tokenizer.json", "onnx/model_quint8_avx2.onnx"]:
            path = target / file
            manifest["files"][file] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        Path("research/experiments/campaign-10", name + "-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "variant": name,
                    "bytes": model.stat().st_size,
                    "seconds": manifest["quantization"]["seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
