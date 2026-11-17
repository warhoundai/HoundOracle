"""Create an unfunded, reviewable Cloud Run manifest. No provisioning or inference."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    data = Path("/workspace/houndoracle-data")
    frozen = json.loads(
        (root / "research/experiments/campaign-04/next-cloud-preflight.json").read_text()
    )
    source = data / "campaign-05-fresh.jsonl"
    if hashlib.sha256(source.read_bytes()).hexdigest() != frozen["dataset_sha256"]:
        raise ValueError("Fresh cohort changed")
    payload = json.loads((data / "cloud-comparison-payload.json").read_text())
    payload["development"] = []
    payload["repetitions"] = 3
    payload["threads"] = 2
    payload["oracle_heads"] = ["logistic"]
    runtime_calibration = (
        root / "research/experiments/campaign-04/calibration-logistic-threads2.json"
    )
    payload["calibrations"]["logistic"] = json.loads(runtime_calibration.read_text())
    payload["confirmation"] = [json.loads(s) for s in source.read_text().splitlines()]
    modules = {"houndoracle/__init__.py": "", "houndoracle/experiments/__init__.py": ""}
    for name in [
        "vertex",
        "nli",
        "head",
        "pipeline",
        "fit_calibration",
        "router",
        "citations",
        "features",
    ]:
        modules[f"houndoracle/experiments/{name}.py"] = (
            root / f"src/houndoracle/experiments/{name}.py"
        ).read_text()
    shared = (root / "scripts/cloud_compare_runner.py").read_text()
    runner = (
        shared[: shared.index("def main():")] + (root / "scripts/cloud_fresh_runner.py").read_text()
    )
    runner = runner.replace("oracle-cloud-03", "oracle-cloud-05")
    # Reuse the previously verified immutable image and existing service account.
    job = json.loads((data / "cloud-run-comparison.json").read_text())
    spec = job["spec"]["template"]["spec"]["template"]["spec"]
    code = """import json,subprocess,sys
from pathlib import Path
subprocess.run(['apt-get','update','-qq'],check=True,stdout=subprocess.DEVNULL)
subprocess.run(['apt-get','install','-y','-qq','--no-install-recommends','libgomp1','libstdc++6'],check=True,stdout=subprocess.DEVNULL)
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','google-genai==2.34.0','onnxruntime==1.32.0','numpy==2.5.4','tokenizers==0.22.2','scipy==1.18.1'],check=True)
"""
    code += "\nMODULES=" + repr(modules) + '\nBASE=Path("/tmp/fresh-app")\n'
    code += "for name,source in MODULES.items():\n p=BASE/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
    code += "sys.path.insert(0,str(BASE))\nPAYLOAD=json.loads(" + repr(json.dumps(payload)) + ")\n"
    code += "exec(" + repr(runner) + ', {"__name__":"__main__","PAYLOAD":PAYLOAD})\n'
    compile(code, "<fresh-cloud-job>", "exec")
    job["metadata"] = {
        "name": "houndoracle-fresh-20261116",
        "labels": {"experiment": "houndoracle-cloud-05"},
    }
    spec["timeoutSeconds"] = "1200"
    spec["containers"][0]["resources"]["limits"]["cpu"] = "2"
    spec["containers"][0]["args"] = ["-c", code]
    (data / "cloud-run-fresh.json").write_text(json.dumps(job, indent=2) + "\n")
    (data / "cloud-fresh-payload.json").write_text(json.dumps(payload, indent=2) + "\n")
    frozen.update(
        authorization_requested_usd=8,
        worker_timeout_seconds=1200,
        script_sha256=hashlib.sha256(code.encode()).hexdigest(),
        job_name=job["metadata"]["name"],
        region="us-central1",
        cpu=2,
        memory="2Gi",
        approval_required_reason="User requires fresh spending authorization; the previous $3 pilot campaign is closed.",
        no_new_iam_grants=True,
        cold_cache=True,
        controls="Identical evidence, model and output format. Same empty-evidence preflight. No answer caching in either arm; local evaluator cache disabled to measure cold performance.",
    )
    (root / "research/experiments/campaign-04/next-cloud-preflight.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in frozen.items() if k != "ids"}, indent=2))


if __name__ == "__main__":
    main()
