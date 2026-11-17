"""Prepare, never provision, a bounded real Gemini comparison against frozen local outputs."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import zlib
from pathlib import Path

from cloud_bundle import validate_bundle
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from houndoracle.experiments.dataset import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorized-usd", type=int, choices=[0, 5], default=0)
    args = parser.parse_args()
    root = Path("/workspace/houndoracle-data")
    public = Path("research/experiments/campaign-14")
    existing = public / "cloud-preflight.json"
    if existing.exists() and json.loads(existing.read_text()).get("status", "").startswith(
        "COMPLETED"
    ):
        raise FileExistsError(
            "Completed campaign is frozen; prepare a separately versioned campaign"
        )
    rows = [
        json.loads(line) for line in (root / "campaign-10/locked.jsonl").read_text().splitlines()
    ]
    selected = sorted(rows, key=lambda r: digest([1930, r["id"]]))[:48]
    payload = {
        "repetitions": 3,
        "questions": [
            {k: r[k] for k in ["id", "group_id", "question", "expected", "evidence"]}
            for r in selected
        ],
    }
    files = {
        "houndoracle/__init__.py": "",
        "houndoracle/experiments/__init__.py": "",
        "runner.py": Path("scripts/cloud_public_gemini_runner.py").read_text(),
    }
    for name in ["vertex", "journal", "execution", "citations"]:
        files[f"houndoracle/experiments/{name}.py"] = Path(
            f"src/houndoracle/experiments/{name}.py"
        ).read_text()
    required = {"google-genai": set()}
    seen = {}
    todo = ["google-genai"]
    while todo:
        name = todo.pop()
        extras = required[name]
        if seen.get(name) == extras:
            continue
        seen[name] = extras.copy()
        for raw_req in importlib.metadata.requires(name) or []:
            req = Requirement(raw_req)
            if req.marker and not any(
                req.marker.evaluate({"extra": extra}) for extra in ["", *extras]
            ):
                continue
            child = canonicalize_name(req.name)
            previous = required.get(child)
            merged = (previous or set()) | set(req.extras)
            if previous is None or merged != previous:
                required[child] = merged
                todo.append(child)
    requirements = (
        "\n".join(sorted(name + "==" + importlib.metadata.version(name) for name in required))
        + "\n"
    )
    files["requirements.txt"] = requirements
    (public / "requirements.txt").write_text(requirements)
    validate_bundle(files)
    bundle = {"files": files, "payload": payload}
    raw = json.dumps(bundle, sort_keys=True).encode()
    encoded = base64.b64encode(zlib.compress(raw)).decode()
    bootstrap = (
        "import base64,json,os,subprocess,sys,zlib\nfrom pathlib import Path\n"
        + f"bundle=json.loads(zlib.decompress(base64.b64decode({encoded!r})))\n"
        + "work=Path('/tmp/oracle-public14');work.mkdir(exist_ok=True);os.chdir(work)\nfor name,body in bundle['files'].items():\n p=Path(name);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)\nPath('payload.json').write_text(json.dumps(bundle['payload']))\nsubprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-cache-dir','-r','requirements.txt'],check=True)\nsubprocess.run([sys.executable,'runner.py','payload.json'],check=True)\n"
    )
    job = {
        "apiVersion": "run.googleapis.com/v1",
        "kind": "Job",
        "metadata": {
            "name": "houndoracle-public-nli-20261116",
            "labels": {"experiment": "houndoracle-public-14"},
        },
        "spec": {
            "template": {
                "spec": {
                    "parallelism": 1,
                    "taskCount": 1,
                    "template": {
                        "spec": {
                            "maxRetries": 0,
                            "timeoutSeconds": "1200",
                            "serviceAccountName": "warhound-gemini@project-f6e3c8ea-6b96-44a8-afa.iam.gserviceaccount.com",
                            "containers": [
                                {
                                    "image": "docker.io/library/python@sha256:1aaa65a85fda306ffb8b910824d4e93bdce61e212c7e87168123ea3073b41a1a",
                                    "command": ["python"],
                                    "args": ["-c", bootstrap],
                                    "env": [
                                        {
                                            "name": "HOUNDORACLE_PILOT_APPROVED_USD",
                                            "value": str(args.authorized_usd),
                                        }
                                    ],
                                    "resources": {"limits": {"cpu": "2", "memory": "2Gi"}},
                                }
                            ],
                        }
                    },
                }
            }
        },
    }
    path = root / "cloud-public-nli-job.json"
    path.write_text(json.dumps(job, indent=2) + "\n")
    (root / "public-gemini-payload.json").write_text(json.dumps(payload) + "\n")
    preflight = {
        "status": "authorized" if args.authorized_usd else "PREPARED_AWAITING_FRESH_AUTHORIZATION",
        "authorized_new_spend_usd": args.authorized_usd,
        "proposed_maximum_new_spend_usd": 5,
        "project": "project-f6e3c8ea-6b96-44a8-afa",
        "job": job["metadata"]["name"],
        "region": "us-central1",
        "model": "gemini-3.8-flash",
        "endpoint": "global",
        "sdk": "2.34.0",
        "thinking": "LOW",
        "seed": 1930,
        "temperature": 0,
        "input_limit": 8192,
        "output_limit": 4096,
        "response_format": "evidence_indices",
        "questions": 48,
        "repetitions": 3,
        "maximum_model_calls": 144,
        "maximum_token_reservations_usd": 3.096576,
        "compute_allowance_usd": 0.50,
        "miscellaneous_allowance_usd": 0.10,
        "billing_and_credits": "Unverified; any approval authorizes pay-as-you-go, not assumed credits. Application reservations are not a universal billing hard cap.",
        "quota": "Cloud Quotas API disabled; no API enabling or IAM changes performed. Capacity availability remains unverified; stop on429.",
        "model_prices": {
            "input_per_million": 0.75,
            "output_and_reasoning_per_million": 3.75,
            "valid_until": "2026-12-31",
            "source": "https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing",
        },
        "dataset_sha256": hashlib.sha256(
            json.dumps(payload["questions"], sort_keys=True).encode()
        ).hexdigest(),
        "ids": [r["id"] for r in selected],
        "class_counts": {
            v: sum(r["expected"] == v for r in selected) for v in ["YES", "NO", "MAYBE"]
        },
        "job_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bundle_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": {
            n: hashlib.sha256(body.encode()).hexdigest() for n, body in files.items()
        },
        "private_job_manifest": str(path),
        "scope": "Controlled matched-evidence Gemini baseline versus independently measured local trained evaluator. No full hybrid/agent latency, live grounding, training weights upload, GPU, public endpoint, bucket or new IAM grants. No automatic retry/resubmission. Delete temporary job after completion.",
        "measurement_limits": "Report real Gemini usage/latency and local evaluator traces separately. Any fallback-policy replay is counterfactual, not measured end-to-end hybrid workflow.",
        "prior_authorizations": "Earlier$3/$8 pilots are closed and not reused.",
    }
    (public / "cloud-preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: preflight[k]
                for k in [
                    "status",
                    "questions",
                    "maximum_model_calls",
                    "maximum_token_reservations_usd",
                    "proposed_maximum_new_spend_usd",
                    "job_manifest_sha256",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
