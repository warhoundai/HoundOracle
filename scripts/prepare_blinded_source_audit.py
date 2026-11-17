"""Trace all selected development inputs upstream, retaining labels and blinding reviewers."""

import hashlib
import json
from pathlib import Path

from houndoracle.experiments.snapshot import wikipedia_revision_reference


def main():
    root = Path("/workspace/houndoracle-data/campaign-32")
    rows = json.loads((root / "inputs.json").read_text())["questions"]
    ids = {r["id"].removeprefix("vitaminc:"): r for r in rows}
    output = root / "blinded-source-audit.jsonl"
    if output.exists():
        raise FileExistsError("Preserve prepared blinded cohort")
    found = {}
    with Path("/workspace/houndoracle-data/vitaminc-sources/dev.jsonl").open() as source:
        for line in source:
            r = json.loads(line)
            if r["unique_id"] not in ids:
                continue
            selected = ids[r["unique_id"]]
            label = {"SUPPORTS": "YES", "REFUTES": "NO", "NOT ENOUGH INFO": "MAYBE"}[r["label"]]
            if (selected["question"], selected["expected"], selected["evidence"][0]["text"]) != (
                r["claim"],
                label,
                r["evidence"],
            ):
                raise ValueError("Transformation mismatch; investigate before scoring")
            found[r["unique_id"]] = {
                "id": selected["id"],
                "group_id": selected["group_id"],
                "claim": r["claim"],
                "evidence": r["evidence"],
                "source_page": r["page"],
                "revision_url": wikipedia_revision_reference(r["page"], r["wiki_revision_id"]),
                "original_record_sha256": hashlib.sha256(
                    json.dumps(r, sort_keys=True).encode()
                ).hexdigest(),
                "upstream_passage_match": "not independently revalidated",
                "review_decision": None,
                "reviewer": None,
                "supporting_spans": [],
            }
    if len(found) != len(rows):
        raise ValueError("Missing upstream records")
    ordered = [found[r["id"].removeprefix("vitaminc:")] for r in rows]
    assert all(
        not any(k in r for k in ["expected", "label", "candidate", "model_decision"])
        for r in ordered
    )
    output.write_text("".join(json.dumps(r) + "\n" for r in ordered))
    public = {
        "rows": len(rows),
        "all_transformed_labels_claims_passages_match_original": True,
        "labels_changed": 0,
        "rows_removed": 0,
        "audit_sampling": "All48preselected cases, not only model errors",
        "review_status": "Pending independent blinded adjudication; no audited labels invented",
        "scope": "Some annotations/context may be ambiguous; model agreement is not proof the gold label is wrong. Existing scores unchanged. Revision references do not establish passage match or objective truth.",
        "records": [
            {k: r[k] for k in ["id", "group_id", "revision_url", "original_record_sha256"]}
            for r in ordered
        ],
    }
    Path("research/experiments/campaign-32/source-audit-preparation.json").write_text(
        json.dumps(public, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in public.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
