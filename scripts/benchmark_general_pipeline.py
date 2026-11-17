"""Measure public API integration on old development cases; never promote calibration."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from houndoracle import DocumentInput, Oracle
from houndoracle.experiments.metrics import paired_comparison, summary
from houndoracle.failures import operational_failure
from houndoracle.general import GeneralVerifier
from houndoracle.local_semantics import LocalPairEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", choices=["44", "45", "44-reproduction"], default="44")
    args = parser.parse_args()
    public = Path("research/experiments") / ("campaign-" + args.campaign)
    public.mkdir(exist_ok=True)
    out = Path("benchmarks/results") / ("campaign-" + args.campaign)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preserve existing per-case traces")
    verifier_class = GeneralVerifier
    if args.campaign == "44-reproduction":
        import importlib.util

        archived = Path("/workspace/houndoracle-data/campaign-44/general-before-v2.py")
        expected = json.loads(Path("research/experiments/campaign-44/runtime.json").read_text())[
            "source_sha256"
        ]["src/houndoracle/general.py"]
        if hashlib.sha256(archived.read_bytes()).hexdigest() != expected:
            raise ValueError("Archived baseline changed")
        spec = importlib.util.spec_from_file_location("houndoracle._baseline44", archived)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        verifier_class = module.GeneralVerifier
    if (public / "screen.json").exists():
        raise FileExistsError("Preserve completed experiment")
    source = Path("/workspace/houndoracle-data/campaign-35/inputs.json")
    rows = json.loads(source.read_text())["validation"]
    if len(rows) != 207:
        raise ValueError("Wrong development cohort")
    protocol = json.loads((public / "protocol.json").read_text())
    model = LocalPairEvaluator(
        "/workspace/houndoracle-models/deberta-vitc-fever-1963",
        expected_weights_sha256=protocol["checkpoint_sha256"],
    )
    systems = {"legacy": [], "general": []}
    for i, row in enumerate(rows):
        for name, output in systems.items():
            oracle = Oracle(general_verifier=verifier_class(model) if name == "general" else None)
            try:
                for j, e in enumerate(row["evidence"]):
                    now = time.time()
                    oracle.store.ingest(
                        DocumentInput(
                            url=e.get("url", f"https://snapshot.example.org/{i}/{j}"),
                            text=e["text"],
                            origin="curator-supplied-snapshot",
                            observed_at=now,
                            expires_at=now + 3600,
                            reviewed=True,
                            primary=False,
                            retrieval_kind="source_document",
                        )
                    )
                started = time.perf_counter()
                result = oracle.verify(row["question"], max_seconds=120)
                trace = result.diagnostics
                output.append(
                    {k: row[k] for k in ["id", "group_id", "expected"]}
                    | {
                        "decision": result.decision.value,
                        "candidate": trace.candidate_decision.value if trace else "MAYBE",
                        "reason": result.reason,
                        "raw_score": trace.raw_score if trace else None,
                        "confidence": result.confidence,
                        "evidence_sufficient": trace.evidence_sufficient if trace else False,
                        "passages": trace.evaluated_passages if trace else 0,
                        "latency_seconds": time.perf_counter() - started,
                        "requests": result.usage["requests"],
                        "failure": operational_failure(result.reason),
                        "citation_supported": None,
                        "all_in_cost_usd": None,
                    }
                )
            finally:
                oracle.store.close()
        if (i + 1) % 25 == 0:
            print(json.dumps({"scored": i + 1, "total": 207}), flush=True)
    report = {
        "source_sha256": hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
        "model_configuration": model.config,
        "accepted_API": {name: summary(rs) for name, rs in systems.items()},
        "unqualified_diagnostic_candidates": {
            name: summary([r | {"decision": r["candidate"]} for r in rs])
            for name, rs in systems.items()
        },
        "paired_candidate_comparison": paired_comparison(
            *[[r | {"decision": r["candidate"]} for r in rs] for rs in systems.values()],
            replicates=2000,
            seed=2001,
        ),
        "new_cloud_spend_usd": 0,
        "qualification95_passed": False,
        "production_qualified": False,
        "scope": "All207olddevelopment examples through actual publicAPI; curated supplied passages only. Candidate metrics are unqualified diagnostic measurements, not accepted accuracy, liveweb performance or a fresh95%gate.",
    }
    (public / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    out.mkdir(parents=True, exist_ok=True)
    for name, rs in systems.items():
        (out / f"{name}-metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
