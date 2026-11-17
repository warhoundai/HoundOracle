"""Collect full public-pipeline candidates; labels never enter model inference."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from houndoracle import DocumentInput, Oracle
from houndoracle.failures import operational_failure
from houndoracle.general import GeneralVerifier
from houndoracle.local_semantics import LocalPairEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["calibration", "validation"])
    args = parser.parse_args()
    public = Path("research/experiments/campaign-47")
    private = Path("/workspace/houndoracle-data/campaign-47")
    if args.stage == "validation":
        frozen = json.loads((public / "frozen-map.json").read_text())
        if not frozen["primary_research_risk_passed"]:
            raise PermissionError("Do not open validation labels after a failed risk check")
    source = Path("/workspace/houndoracle-data/qualification95-candidates") / (
        args.stage + ".jsonl"
    )
    manifest = json.loads(Path("research/qualification95/candidate-manifest.json").read_text())
    if (
        hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
        != manifest["panels"][args.stage]["sha256"]
    ):
        raise ValueError("Reserved source changed")
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    if len(rows) != (1000 if args.stage == "calibration" else 500) or len(
        {r["group_id"] for r in rows}
    ) != len(rows):
        raise ValueError("Cohort dimensions or dependence changed")
    target = private / (args.stage + "-scores.jsonl")
    if target.exists():
        raise FileExistsError("Preserve scored cohort")
    model = LocalPairEvaluator(
        "/workspace/houndoracle-models/deberta-vitc-fever-1963",
        expected_weights_sha256="c520871a91b6497b628d60aad945b54c34819d8b53c5c3b0bf814fab7d3f3e70",
    )
    identity = Oracle(general_verifier=GeneralVerifier(model))
    from houndoracle.models import VerifyRequest

    policy = identity.general_verifier.request_policy_id(
        identity, VerifyRequest(question="Reference proposition", max_seconds=120)
    )
    identity.store.close()
    if args.stage == "validation" and frozen["policy_id"] != policy:
        raise ValueError("Frozen model/request policy changed")
    with target.open("x") as output:
        for i, r in enumerate(rows):
            oracle = Oracle(general_verifier=GeneralVerifier(model))
            try:
                for e in r["evidence"]:
                    now = time.time()
                    oracle.store.ingest(
                        DocumentInput(
                            url=e["url"],
                            text=e["text"],
                            origin="curator-supplied-public-annotation",
                            observed_at=now,
                            expires_at=now + 3600,
                            reviewed=True,
                            retrieval_kind="source_document",
                        )
                    )
                result = oracle.verify(r["question"], max_seconds=120)
                trace = result.diagnostics
                record = {k: r[k] for k in ["id", "group_id", "expected"]} | {
                    "decision": result.decision.value,
                    "candidate": trace.candidate_decision.value if trace else "MAYBE",
                    "raw_score": trace.raw_score if trace else None,
                    "failure": operational_failure(result.reason),
                    "reason": result.reason,
                    "policy_id": result.versions["policy"],
                    "confidence": result.confidence,
                    "latency_seconds": result.usage["latency_seconds"],
                    "truncated": False,
                    "citation_supported": None,
                }
                if record["failure"] or record["policy_id"] != policy:
                    record["candidate"] = "MAYBE"
                output.write(json.dumps(record) + "\n")
            finally:
                oracle.store.close()
            if (i + 1) % 50 == 0:
                output.flush()
                print(
                    json.dumps({"stage": args.stage, "scored": i + 1, "total": len(rows)}),
                    flush=True,
                )
    (private / (args.stage + "-runtime.json")).write_text(
        json.dumps(
            {
                "policy_id": policy,
                "model_config": model.config,
                "source_sha256": manifest["panels"][args.stage]["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
