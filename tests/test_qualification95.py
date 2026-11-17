import pytest

from houndoracle.experiments.qualification95 import assess_track, validate_partitions


def fixture():
    gold = [
        {"id": str(i), "expected": "YES" if i < 400 else "NO" if i < 800 else "MAYBE"}
        for i in range(1000)
    ]
    results = [
        r
        | {
            "decision": r["expected"],
            "policy_id": "frozen",
            "confidence": 0.99 if i < 800 else None,
            "citation_supported": True,
            "citation_audit_method": "independent_span_review",
        }
        for i, r in enumerate(gold)
    ]
    return gold, results


def test_exact95_boundary_and_failures_never_become_correct_maybe():
    gold, rows = fixture()
    for i in range(50):
        rows[800 + i]["failure"] = "NETWORK_FAILED"
    score = assess_track(gold, rows, policy_id="frozen")
    assert score["accuracy"] == 0.95 and score["failures"] == 50 and score["gate_passed"]
    rows[850]["failure"] = "NETWORK_FAILED"
    assert not assess_track(gold, rows, policy_id="frozen")["gate_passed"]


def test_abstention_and_unaudited_citations_cannot_manufacture_pass():
    gold, rows = fixture()
    for r in rows:
        r["decision"] = "MAYBE"
        r["confidence"] = None
    score = assess_track(gold, rows, policy_id="frozen")
    assert score["accuracy"] == 0.2 and not score["gate_passed"]
    gold, rows = fixture()
    for r in rows:
        r["citation_audit_method"] = "model_self_review"
    assert not assess_track(gold, rows, policy_id="frozen")["gate_passed"]
    for r in rows:
        r["confidence"] = None
    assert assess_track(gold, rows, policy_id="frozen")["failures"] == 800


def test_case_denominator_and_policy_immutability():
    gold, rows = fixture()
    with pytest.raises(ValueError):
        assess_track(gold, rows[:-1], policy_id="frozen")
    with pytest.raises(ValueError):
        assess_track(gold, rows[:-1] + [rows[0]], policy_id="frozen")
    rows[0]["policy_id"] = "tuned-after-test"
    with pytest.raises(ValueError):
        assess_track(gold, rows, policy_id="frozen")


def test_partition_validator_rejects_silver_or_unaudited_data():
    row = {"id": "a", "label_origin": "generated", "synthetic": True}
    parts = {
        track: {
            split: [dict(row, id=f"{track}-{split}")]
            for split in ["development", "validation", "calibration", "locked"]
        }
        for track in ["passage", "end_to_end"]
    }
    with pytest.raises(ValueError, match="nonsynthetic"):
        validate_partitions(parts)
    for splits in parts.values():
        for rows in splits.values():
            rows[0].update(label_origin="human_dataset", synthetic=False)
    with pytest.raises(ValueError, match="audit"):
        validate_partitions(parts)


def audited_fixture_partitions():
    from houndoracle.experiments.qualification95 import CATEGORIES

    result = {}
    for track in ["passage", "end_to_end"]:
        splits = {}
        for split, n in [
            ("development", 1),
            ("validation", 500),
            ("calibration", 1000),
            ("locked", 1000),
        ]:
            rows = []
            for i in range(n):
                key = f"unit-{track}-{split}-{i}"
                rows.append(
                    {
                        "id": key,
                        "question": key,
                        "group_id": key,
                        "event_id": key,
                        "source_ids": ["fixture:" + key],
                        "expected": "YES" if i < n * 0.4 else "NO" if i < n * 0.8 else "MAYBE",
                        "label_origin": "human_dataset",
                        "synthetic": False,
                        "audit_status": "verified",
                        "audit_reference": "unit-fixture-not-real-review",
                        "as_of": "2020-01-01",
                        "evidence_scope": track,
                        "annotation_reference": "fixture-only",
                        "commercial_use_reviewed": True,
                        "license_reference": "fixture-only",
                        "categories": list(CATEGORIES),
                        "domain": str(i % 5),
                    }
                )
            splits[split] = rows
        result[track] = splits
    return result


def test_partition_graph_leakage_and_web_label_scope():
    parts = audited_fixture_partitions()
    assert validate_partitions(parts)["partitions"]["passage:locked"]["count"] == 1000
    parts["end_to_end"]["locked"][0]["event_id"] = parts["passage"]["development"][0]["event_id"]
    with pytest.raises(ValueError, match="leak"):
        validate_partitions(parts)
    parts = audited_fixture_partitions()
    parts["end_to_end"]["locked"][900]["label_basis"] = "passage_only_nei"
    with pytest.raises(ValueError, match="NEI"):
        validate_partitions(parts)


def test_failed_validation_never_opens_locked_file(tmp_path, monkeypatch):
    import json
    import runpy
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/qualification95.py"
    main = runpy.run_path(str(script))["main"]
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "policy_id": "p",
                "tracks": {
                    "passage": {"accuracy": 0.96, "count": 500},
                    "end_to_end": {"accuracy": 0.99, "count": 500},
                },
            }
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "qualification95",
            "score",
            "--validation",
            str(validation),
            "--policy-id",
            "p",
            "--partitions",
            str(tmp_path / "missing-locked.json"),
            "--results",
            str(tmp_path / "missing-results.json"),
            "--registry",
            str(tmp_path),
        ],
    )
    with pytest.raises(PermissionError, match="validationgates"):
        main()


def test_missing_failure_flag_does_not_hide_operational_maybe():
    gold, rows = fixture()
    for r in rows[800:]:
        r["reason"] = "RETRIEVAL_FAILED"
    score = assess_track(gold, rows, policy_id="frozen")
    assert score["correct"] == 800 and score["failures"] == 200
    assert not score["gate_passed"]
