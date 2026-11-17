import time

from houndoracle.calibration import GeneralCalibration, load_calibration


def artifact(**kw):
    return GeneralCalibration(
        policy_id="exact-policy",
        population="synthetic-test-only",
        supported_kinds=["prose", "not"],
        expires_at=time.time() + 100,
        dataset_digest="0" * 64,
        weight=1,
        bias=0,
        risk_lower_bound=0.9,
        risk_groups=100,
        **kw,
    )


def test_research_calibration_cannot_authorize_serving_even_with_good_metrics():
    a = artifact()
    assert abs(a.estimate(0.95) - 0.95) < 1e-10
    assert a.confidence(0.99, {"prose"}, "exact-policy") is None


def test_audited_map_is_conservative_and_strictly_scoped():
    a = artifact(qualification_status="audited", audit_references=["unit-fixture-not-real-audit"])
    assert a.confidence(0.99, {"prose"}, "exact-policy") == 0.9
    assert a.confidence(0.79, {"prose"}, "exact-policy") is None
    assert a.confidence(0.99, {"quantity"}, "exact-policy") is None
    assert a.confidence(0.99, {"prose"}, "different-policy") is None
    assert a.confidence(float("nan"), {"prose"}, "exact-policy") is None
    assert a.estimate(True) is None
    a.expires_at = time.time() - 1
    assert a.confidence(0.99, {"prose"}, "exact-policy") is None


def test_legacy_loader_distinguishes_general_research_schema(tmp_path):
    path = tmp_path / "research-calibration.json"
    a = artifact()
    path.write_text(a.model_dump_json())
    loaded = load_calibration(path)
    assert loaded.version == a.version and loaded.qualification_status == "research"
    assert loaded.confidence(0.99, {"prose"}, "exact-policy") is None


def test_research_artifact_does_not_make_api_look_calibrated():
    from fastapi.testclient import TestClient

    from houndoracle import Oracle
    from houndoracle.api import create_app

    oracle = Oracle(calibration=artifact())
    try:
        with TestClient(
            create_app(oracle, api_key="unit-instance-key-with-at-least24chars")
        ) as client:
            health = client.get("/health").json()
            assert health["calibrated"] is False and health["calibration_status"] == "research"
    finally:
        oracle.store.close()
