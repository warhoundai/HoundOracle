"""Build a private bounded job; signed model URL never enters public artifacts."""

import base64
import hashlib
import importlib.metadata
import json
import zlib
from pathlib import Path

from cloud_bundle import validate_bundle


def main():
    data = Path("/workspace/houndoracle-data")
    public = Path("research/experiments/campaign-18")
    payload = json.loads((data / "public-gemini-payload.json").read_text())
    frozen = [
        json.loads(l)
        for l in Path("benchmarks/results/campaign-14/local-records.jsonl").read_text().splitlines()
    ]
    payload |= {
        "manifest": json.loads(
            Path("research/experiments/campaign-08/round-2/trained-manifest.json").read_text()
        ),
        "head": json.loads(
            Path("research/experiments/campaign-09/continued-mixture-head.json").read_text()
        ),
        "calibration": json.loads(
            Path("research/experiments/campaign-09/continued-mixture-calibration.json").read_text()
        ),
        "frozen_local_decisions": {r["id"]: r["decision"] for r in frozen},
        "dataset_sha256": hashlib.sha256(
            json.dumps(payload["questions"], sort_keys=True).encode()
        ).hexdigest(),
    }
    files = {
        "houndoracle/__init__.py": "",
        "houndoracle/experiments/__init__.py": "",
        "runner.py": Path("scripts/cloud_hybrid_workflow_runner.py").read_text(),
    }
    for name in [
        "vertex",
        "citations",
        "journal",
        "execution",
        "head",
        "nli",
        "pipeline",
        "fit_calibration",
        "features",
    ]:
        files[f"houndoracle/experiments/{name}.py"] = Path(
            f"src/houndoracle/experiments/{name}.py"
        ).read_text()
    requirements = Path("research/experiments/campaign-14/requirements.txt").read_text()
    # Pin the actual tested transitive inference libraries, without PyTorch/sklearn/model code execution.
    for name in [
        "onnxruntime",
        "numpy",
        "scipy",
        "tokenizers",
        "flatbuffers",
        "protobuf",
        "packaging",
    ]:
        requirements += f"{name}=={importlib.metadata.version(name)}\n"
    requirements = "\n".join(sorted(set(requirements.splitlines()))) + "\n"
    files["requirements.txt"] = requirements
    (public / "requirements.txt").write_text(requirements)
    validate_bundle(files)
    bundle = {"files": files, "payload": payload}
    raw = json.dumps(bundle, sort_keys=True).encode()
    encoded = base64.b64encode(zlib.compress(raw)).decode()
    transfer = json.loads((data / "workflow18-transfer-urls.json").read_text())
    sha = hashlib.sha256((data / "workflow18-model.tar.gz").read_bytes()).hexdigest()
    bootstrap = "import base64,hashlib,json,os,subprocess,sys,tarfile,urllib.request,zlib\nfrom pathlib import Path\n"
    bootstrap += f"bundle=json.loads(zlib.decompress(base64.b64decode({encoded!r})))\n"
    bootstrap += "work=Path('/tmp/workflow18');work.mkdir(exist_ok=True);os.chdir(work)\nfor name,body in bundle['files'].items():\n p=Path(name);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)\nPath('payload.json').write_text(json.dumps(bundle['payload']))\n"
    bootstrap += f"with urllib.request.urlopen({transfer['download_url']!r},timeout=90) as source,open('/tmp/model.tar.gz','wb') as target:\n size=0\n while chunk:=source.read(1048576):\n  size+=len(chunk)\n  if size>340000000:raise ValueError('Model archive exceeds envelope')\n  target.write(chunk)\nassert hashlib.sha256(Path('/tmp/model.tar.gz').read_bytes()).hexdigest()=={sha!r}\n"
    bootstrap += "Path('/tmp/model').mkdir(exist_ok=True)\nwith tarfile.open('/tmp/model.tar.gz') as archive:archive.extractall('/tmp/model',filter='data')\nPath('/tmp/model.tar.gz').unlink()\nsubprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-cache-dir','-r','requirements.txt'],check=True)\nsubprocess.run([sys.executable,'runner.py'],check=True)\n"
    compile(bootstrap, "<workflow18>", "exec")
    job = json.loads((data / "cloud-public-nli-job.json").read_text())
    job["metadata"]["name"] = "houndoracle-hybrid-workflow-20261116"
    job["metadata"]["labels"]["experiment"] = "houndoracle-workflow-18"
    spec = job["spec"]["template"]["spec"]["template"]["spec"]
    spec["timeoutSeconds"] = "1800"
    spec["containers"][0]["args"] = ["-c", bootstrap]
    spec["containers"][0].pop("env", None)
    target = data / "cloud-workflow18-job.json"
    target.write_text(json.dumps(job, indent=2) + "\n")
    (data / "workflow18-payload.json").write_text(json.dumps(payload) + "\n")
    report = {
        "status": "AUTHORIZED_PREPARED",
        "job": job["metadata"]["name"],
        "region": "us-central1",
        "gcp_reserved_usd": 8,
        "aws_transfer_reserved_usd": 2,
        "maximum_model_calls": 288,
        "maximum_model_reservations_usd": 6.193152,
        "seconds": 1800,
        "application_seconds": 1500,
        "dataset_sha256": payload["dataset_sha256"],
        "model_archive_sha256": sha,
        "model_archive_bytes": (data / "workflow18-model.tar.gz").stat().st_size,
        "job_manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()
        },
        "signed_url_public": False,
        "signed_url_expiry_seconds": 900,
        "production_qualified": False,
    }
    (public / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
