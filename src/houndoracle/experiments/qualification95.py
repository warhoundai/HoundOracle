"""Frozen two-track acceptance contract. No test-set tuning or synthetic substitution."""

import hashlib
import json
import math
from collections import Counter

from ..failures import operational_failure

TRACKS = {"passage", "end_to_end"}
SPLITS = {"development", "validation", "calibration", "locked"}
CATEGORIES = {
    "technical",
    "nontechnical",
    "dates",
    "quantities",
    "negation",
    "comparisons",
    "multi_hop",
    "contradictory",
    "ambiguous",
    "adversarial",
}


CATEGORY_MINIMUMS = {
    category: (200 if category in {"technical", "nontechnical"} else 100) for category in CATEGORIES
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_partitions(partitions):
    """Validate curator-supplied provenance; cannot manufacture independent audits."""
    if set(partitions) != TRACKS:
        raise ValueError("Both evaluation tracks required")
    owner = {}
    ids = set()
    manifests = {}
    for track, splits in partitions.items():
        if set(splits) != SPLITS:
            raise ValueError("Four separate partitions required")
        for split, rows in splits.items():
            if not rows:
                raise ValueError("Empty partition")
            for r in rows:
                if r["id"] in ids:
                    raise ValueError("Duplicate case identity")
                ids.add(r["id"])
                if (
                    r.get("label_origin") not in {"human_dataset", "independent_human"}
                    or r.get("synthetic") is not False
                ):
                    raise ValueError("Independent nonsynthetic labels required")
                if r.get("audit_status") != "verified" or not r.get("audit_reference"):
                    raise ValueError("Blind source audit required")
                if r.get("expected") not in {"YES", "NO", "MAYBE"}:
                    raise ValueError("Valid three-way label required")
                if (
                    not r.get("as_of")
                    or not r.get("evidence_scope")
                    or not r.get("annotation_reference")
                ):
                    raise ValueError("Temporal and annotation provenance required")
                if r.get("commercial_use_reviewed") is not True or not r.get("license_reference"):
                    raise ValueError("Dataset-specific licensing review required")
                if track == "end_to_end" and r.get("label_basis") == "passage_only_nei":
                    raise ValueError("Passage NEI is not open-web ground truth")
                if not r.get("group_id") or not r.get("source_ids") or not r.get("event_id"):
                    raise ValueError("Source/event component identities required")
                keys = [
                    "group:" + r["group_id"],
                    "event:" + r["event_id"],
                    "claim:" + digest(" ".join(r["question"].casefold().split())),
                ] + ["source:" + v for v in r["source_ids"]]
                for key in keys:
                    if key in owner and owner[key] != (track, split):
                        raise ValueError("Related components leak across partitions")
                    owner[key] = (track, split)
            manifests[f"{track}:{split}"] = {"count": len(rows), "sha256": digest(rows)}
        if len(splits["validation"]) < 500 or len(splits["calibration"]) < 1000:
            raise ValueError("Validation500andcalibration1000minimum required")
        locked = splits["locked"]
        if len(locked) != 1000 or Counter(r["expected"] for r in locked) != Counter(
            {"YES": 400, "NO": 400, "MAYBE": 200}
        ):
            raise ValueError("Locked1000case400/400/200 contract required")
        categories = set().union(*(set(r.get("categories", [])) for r in locked))
        if not CATEGORIES <= categories:
            raise ValueError("Missing required difficulty categories")
        category_counts = Counter(
            category for row in locked for category in set(row.get("categories", []))
        )
        if any(category_counts[k] < n for k, n in CATEGORY_MINIMUMS.items()):
            raise ValueError("Required hard-case category coverage not met")
        if max(Counter(r.get("domain", "unspecified") for r in locked).values()) > 200:
            raise ValueError("Narrow domain exceeds20percent")
    return {
        "schema_version": 1,
        "partitions": manifests,
        "target_observed_accuracy": 0.95,
        "dataset_digest": digest(manifests),
        "test_predictions_opened": False,
    }


def assess_track(expected, results, *, policy_id):
    if len(expected) != 1000 or len({r["id"] for r in expected}) != 1000:
        raise ValueError("Exactly1000unique frozen cases required")
    by = {r["id"]: r for r in results}
    if len(by) != len(results) or set(by) != {r["id"] for r in expected}:
        raise ValueError("Missing or duplicate outcomes; no successful-prefix scoring")
    correct = 0
    accepted = []
    answerable = 0
    answerable_covered = 0
    failures = 0
    for gold in expected:
        r = by[gold["id"]]
        if r.get("policy_id") != policy_id:
            raise ValueError("Candidate policy changed during evaluation")
        if r.get("decision") not in {"YES", "NO", "MAYBE"}:
            raise ValueError("Invalid decision encoding")
        confidence = r.get("confidence")
        definitive = r["decision"] in {"YES", "NO"}
        valid_confidence = (
            isinstance(confidence, (float, int))
            and not isinstance(confidence, bool)
            and math.isfinite(confidence)
            and 0.8 <= confidence <= 1
        )
        failed = (
            bool(r.get("failure") or operational_failure(r.get("reason")))
            or definitive
            and not valid_confidence
        )
        failures += int(failed)
        correct += int(not failed and r["decision"] == gold["expected"])
        if gold["expected"] in {"YES", "NO"}:
            answerable += 1
            answerable_covered += int(definitive and not failed)
        if definitive and not failed:
            accepted.append((gold, r))
    correct_def = sum(g["expected"] == r["decision"] for g, r in accepted)
    supported = sum(
        r.get("citation_supported") is True
        and r.get("citation_audit_method") == "independent_span_review"
        for _, r in accepted
    )
    precision = correct_def / len(accepted) if accepted else 0.0
    coverage = answerable_covered / answerable if answerable else 0.0
    citations = supported / len(accepted) if accepted else 0.0
    return {
        "count": 1000,
        "correct": correct,
        "accuracy": correct / 1000,
        "failures": failures,
        "definitive_count": len(accepted),
        "correct_definitive": correct_def,
        "definitive_precision": precision,
        "answerable_coverage": coverage,
        "citation_correctness": citations,
        "gate_passed": correct >= 950
        and precision >= 0.95
        and coverage >= 0.95
        and citations >= 0.95,
        "policy_id": policy_id,
        "uncertainty_note": "Observed gate as requested; intervals reported separately, no95percentpopulation guarantee.",
    }


def assess_campaign(
    expected, results, *, policy_id, manifest, validation_accuracy, critical_security_failures=0
):
    if set(expected) != TRACKS or set(results) != TRACKS:
        raise ValueError("Both tracks required; no substitute task")
    if any(
        type(validation_accuracy.get(t)) not in (int, float)
        or not math.isfinite(validation_accuracy[t])
        or not 0.97 <= validation_accuracy[t] <= 1
        for t in TRACKS
    ):
        raise PermissionError("Both validation gates must pass97percent before locked scoring")
    if manifest.get("dataset_digest") != digest(manifest.get("partitions", {})):
        raise ValueError("Sealed manifest changed")
    for track in TRACKS:
        if manifest["partitions"][f"{track}:locked"]["sha256"] != digest(expected[track]):
            raise ValueError("Locked labels changed")
    reports = {
        t: assess_track(expected[t], results[t], policy_id=policy_id) for t in sorted(TRACKS)
    }
    return {
        "tracks": reports,
        "gate_passed": all(r["gate_passed"] for r in reports.values())
        and critical_security_failures == 0,
        "critical_security_failures": critical_security_failures,
        "public_release_authorized": False,
    }
