"""Prepare a reviewable, bounded Vertex CustomJob; does not submit it."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(data_directory, output):
    rows = [
        json.loads(line)
        for line in (Path(data_directory) / "validation.jsonl").read_text().splitlines()
    ]
    selected, groups = [], set()
    for label in ["YES", "NO", "MAYBE"]:
        for row in rows:
            if row["expected"] == label and row["group_id"] not in groups:
                selected.append(row)
                groups.add(row["group_id"])
                if sum(r["expected"] == label for r in selected) == 4:
                    break
    rows = selected
    examples = [{k: r[k] for k in ["id", "question", "evidence"]} for r in rows]
    # Embed only the public evidence, integration source and configuration. Never credentials/gold labels.
    module = (ROOT / "src/houndoracle/experiments/vertex.py").read_text()
    script = (
        """import subprocess,sys,tempfile,json,time
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--quiet','google-genai==2.34.0'],check=True)
"""
        + module
        + "\nEXAMPLES="
        + repr(examples)
        + """
ledger=SpendLedger('/tmp/oracle-pilot-ledger.sqlite',authorized_usd=2.50)
for repeat in range(2):
 for level in ['LOW','MEDIUM']:
  try:
   model=VertexGemini(VertexConfig(project='project-f6e3c8ea-6b96-44a8-afa',thinking_level=level),ledger)
   for row in EXAMPLES:
    try:
     result=model.answer(row['question'],row['evidence'])
     print(json.dumps({'campaign':'oracle-pilot-01','id':row['id'],'repeat':repeat,'thinking':level,'result':result}),flush=True)
    except Exception as exc:
     print(json.dumps({'campaign':'oracle-pilot-01','id':row['id'],'repeat':repeat,'thinking':level,'error_type':type(exc).__name__}),flush=True)
     raise
  except Exception:
   ledger.close()
   raise SystemExit(2)
ledger.close()
"""
    )
    config = {
        "displayName": "houndoracle-matched-pilot-01",
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "machineSpec": {"machineType": "n1-standard-4"},
                    "replicaCount": "1",
                    "containerSpec": {
                        "imageUri": "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6@sha256:a2fea4ad3379535f2e93126bcf06c9ef6dc5803ac42f36fa4af11e1ef8cd3b14",
                        "command": ["python3"],
                        "args": ["-c", script],
                    },
                }
            ],
            "scheduling": {
                "timeout": "600s",
                "restartJobOnWorkerRestart": False,
                "disableRetries": True,
            },
        },
    }
    Path(output).write_text(json.dumps(config["jobSpec"], indent=2) + "\n")
    return {
        "questions": len(examples),
        "model_calls_max": len(examples) * 4,
        "thinking": ["LOW", "MEDIUM"],
        "project": "project-f6e3c8ea-6b96-44a8-afa",
        "model": "gemini-3.8-flash",
        "location": "global",
        "model_reservations_upper_usd": len(examples) * 4 * (8192 * 0.75 + 4096 * 3.75) / 1e6,
        "cpu_allowance_usd": 0.25,
        "billing_status": "unverified",
        "credits": "unverified",
        "submission_status": "not_submitted_requires_new_spending_authorization",
        "no_grounding_in_this_pilot": True,
        "job_timeout_seconds": 600,
        "notes": [
            "No GPU, public endpoint, persistent bucket or automatic retry.",
            "Container image is pinned to the observed immutable registry digest.",
            "The model reservation is not a universal cloud-billing hard cap.",
        ],
    }


if __name__ == "__main__":
    result = build(
        "/workspace/houndoracle-data/campaign-02",
        "/workspace/houndoracle-data/vertex-pilot-job.json",
    )
    (ROOT / "research/cloud/pilot-preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
