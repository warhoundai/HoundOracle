"""Prepare a temporary Cloud Run Job, exact public-data cohorts and embedded source; never submits it."""

import json
from pathlib import Path

from houndoracle.experiments.dataset import digest, load_checked

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "project-f6e3c8ea-6b96-44a8-afa"
NAME = "houndoracle-compare-20261115-a"


def cohorts(*, supplemental=False):
    data = Path("/workspace/houndoracle-data/campaign-02")
    rows = [json.loads(s) for s in (data / "validation.jsonl").read_text().splitlines()]
    dev = []
    used = set()
    for label in ["YES", "NO", "MAYBE"]:
        for row in rows:
            if row["expected"] == label and row["group_id"] not in used:
                dev.append(row)
                used.add(row["group_id"])
                if sum(r["expected"] == label for r in dev) == 2:
                    break
    # Independent confirmation excludes entire components containing any earlier assigned/scored question.
    all_rows, _ = load_checked("/workspace/houndoracle-data/fever-sources")
    parents = {}

    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]

    for row in all_rows:
        for key in ["page:" + p for p in row["source_pages"]] + [
            "subject:" + row["subject"],
            "claim:" + row["question"].casefold(),
        ]:
            parents[find(row["id"])] = find(key)
    groups = {}
    for row in all_rows:
        groups.setdefault(find(row["id"]), []).append(row)
    forbidden = set()
    for path in data.glob("*.jsonl"):
        forbidden.update(json.loads(s)["id"] for s in path.read_text().splitlines())
    for path in Path("/workspace/houndoracle-data/scores-invalidated-v1").glob("*.jsonl"):
        forbidden.update(json.loads(s)["id"] for s in path.read_text().splitlines())
    available = [g for g in groups.values() if not any(r["id"] in forbidden for r in g)]
    if supplemental:
        previous = json.loads(
            Path("/workspace/houndoracle-data/cloud-comparison-payload.json").read_text()
        )
        prior_ids = {r["id"] for part in ["development", "confirmation"] for r in previous[part]}
        available = [g for g in available if not any(r["id"] in prior_ids for r in g)]
    available.sort(key=lambda g: digest([1903, sorted(r["id"] for r in g)]))
    if supplemental:
        fresh = []
        for group in available:
            rows = sorted(group, key=lambda r: digest([1904, r["id"]]))
            row = next((r for r in rows if sum(len(e["text"]) for e in r["evidence"]) < 5000), None)
            if row is not None:
                row["group_id"] = digest(sorted(r["id"] for r in group))
                fresh.append(
                    {k: row[k] for k in ["id", "group_id", "question", "expected", "evidence"]}
                )
            if len(fresh) == 2:
                return [], fresh
        raise ValueError("Two fresh source-disjoint cases unavailable")
    confirmation = []
    selected_groups = set()
    for label in ["YES", "NO", "MAYBE", "YES"]:
        found = False
        for group in available:
            key = digest(sorted(r["id"] for r in group))
            if key in selected_groups:
                continue
            for row in sorted(group, key=lambda r: digest([1903, r["id"]])):
                if row["expected"] == label and sum(len(e["text"]) for e in row["evidence"]) < 5000:
                    row["group_id"] = key
                    confirmation.append(row)
                    selected_groups.add(key)
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError("Fresh source-disjoint confirmation unavailable")
    fields = ["id", "group_id", "question", "expected", "evidence"]
    return [{k: r[k] for k in fields} for r in dev], [
        {k: r[k] for k in fields} for r in confirmation
    ]


def build(*, supplemental=False):
    dev, confirmation = cohorts(supplemental=supplemental)
    payload = {
        "project": PROJECT,
        "development": dev,
        "confirmation": confirmation,
        "model_manifest": json.loads(
            (ROOT / "research/model-comparisons/minilm-provenance.json").read_text()
        ),
        "heads": json.loads(
            (ROOT / "research/experiments/results/trained-aggregators.json").read_text()
        ),
        "calibrations": {
            name: json.loads(
                (ROOT / f"research/experiments/results/calibration-{name}.json").read_text()
            )
            for name in ["mlp", "logistic"]
        },
    }
    modules = {"houndoracle/__init__.py": "", "houndoracle/experiments/__init__.py": ""}
    for name in ["vertex", "nli", "head", "pipeline", "fit_calibration", "router", "citations"]:
        modules[f"houndoracle/experiments/{name}.py"] = (
            ROOT / f"src/houndoracle/experiments/{name}.py"
        ).read_text()
    runner = (ROOT / "scripts/cloud_compare_runner.py").read_text()
    if supplemental:
        runner = (
            runner[: runner.index("def main():")]
            + (ROOT / "scripts/cloud_compact_runner.py").read_text()
        )
    code = """import json,os,subprocess,sys,time
started=time.perf_counter()
print(json.dumps({'campaign':'oracle-cloud-03','kind':'container_started'}),flush=True)
subprocess.run(['apt-get','update','-qq'],check=True)
subprocess.run(['apt-get','install','-y','-qq','--no-install-recommends','libgomp1','libstdc++6'],check=True)
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--disable-pip-version-check','google-genai==2.34.0','onnxruntime==1.32.0','numpy==2.5.4','tokenizers==0.22.2','scipy==1.18.1'],check=True)
print(json.dumps({'campaign':'oracle-cloud-03','kind':'packages_ready','seconds':time.perf_counter()-started}),flush=True)
"""
    code += (
        '\nfrom pathlib import Path\nBASE=Path("/tmp/oracle-app")\nMODULES=' + repr(modules) + "\n"
    )
    code += "for name,source in MODULES.items():\n p=BASE/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
    code += "sys.path.insert(0,str(BASE))\nPAYLOAD=json.loads(" + repr(json.dumps(payload)) + ")\n"
    code += "exec(" + repr(runner) + ', {"__name__":"__main__","PAYLOAD":PAYLOAD})\n'
    if supplemental:
        code = code.replace("oracle-cloud-03", "oracle-cloud-03b")
    compile(code, "<cloud-comparison>", "exec")
    job = {
        "apiVersion": "run.googleapis.com/v1",
        "kind": "Job",
        "metadata": {"name": NAME, "labels": {"experiment": "houndoracle-cloud-03"}},
        "spec": {
            "template": {
                "spec": {
                    "parallelism": 1,
                    "taskCount": 1,
                    "template": {
                        "spec": {
                            "maxRetries": 0,
                            "timeoutSeconds": "600",
                            "serviceAccountName": f"warhound-gemini@{PROJECT}.iam.gserviceaccount.com",
                            "containers": [
                                {
                                    "image": "docker.io/library/python@sha256:1aaa65a85fda306ffb8b910824d4e93bdce61e212c7e87168123ea3073b41a1a",
                                    "command": ["python3"],
                                    "args": ["-c", code],
                                    "resources": {"limits": {"cpu": "1", "memory": "2Gi"}},
                                }
                            ],
                        }
                    },
                }
            }
        },
    }
    suffix = "-supplemental" if supplemental else ""
    if supplemental:
        job["spec"]["template"]["spec"]["template"]["spec"]["timeoutSeconds"] = "180"
    Path(f"/workspace/houndoracle-data/cloud-run-comparison{suffix}.json").write_text(
        json.dumps(job, indent=2) + "\n"
    )
    Path(f"/workspace/houndoracle-data/cloud-comparison-payload{suffix}.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    record = {
        "project": PROJECT,
        "job": NAME,
        "development_ids": [r["id"] for r in dev],
        "confirmation_ids": [r["id"] for r in confirmation],
        "cohort_digest": digest([dev, confirmation]),
        "script_sha256": __import__("hashlib").sha256(code.encode()).hexdigest(),
        "maximum_model_attempts": 48,
        "maximum_comparison_attempts": 12 if supplemental else 40,
        "maximum_smoke_attempts": 0 if supplemental else 4,
        "previous_model_attempts": 35 if supplemental else 0,
        "new_authorized_usd": 3,
        "timeout_seconds": 180 if supplemental else 600,
        "cpu": 1,
        "memory": "2Gi",
        "region": "us-central1",
        "no_grounding": True,
        "new_iam_grants": False,
    }
    (ROOT / f"research/experiments/campaign-03/frozen-job{suffix}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    import sys

    build(supplemental="--supplemental" in sys.argv)
