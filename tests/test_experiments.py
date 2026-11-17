import json
from types import SimpleNamespace
from typing import ClassVar

import pytest

from houndoracle.experiments.dataset import load_checked, partition
from houndoracle.experiments.fit_calibration import fit, predict
from houndoracle.experiments.head import infer_head
from houndoracle.experiments.metrics import paired_comparison, summary
from houndoracle.experiments.pipeline import ResearchPipeline, apply
from houndoracle.experiments.vertex import SpendLedger, VertexConfig, VertexGemini


def test_trace_does_not_promote_uncalibrated(oracle, ingest):
    ingest()
    oracle.calibration = None
    result = oracle.verify("Was Widget released in 2024?")
    assert result.decision == "MAYBE" and result.confidence is None
    assert result.diagnostics.candidate_decision == "YES"
    assert result.diagnostics.evidence_sufficient
    assert 0 <= result.diagnostics.raw_score <= 1


def test_dataset_deduplicates_evidence_variants(tmp_path):
    original = {"id": 1, "claim": "Claim", "label": "SUPPORTS", "evidence": [[[1, 1, "Page", 0]]]}
    (tmp_path / "original.jsonl").write_text(json.dumps(original) + "\n")
    mirror = {
        "original_id": 1,
        "claim": "Claim",
        "label": "SUPPORTS",
        "evidence": [["Page", "0", "Evidence."]],
    }
    second = {**mirror, "evidence": mirror["evidence"] + [["Other", "0", "More evidence."]]}
    (tmp_path / "gold.jsonl").write_text(json.dumps(mirror) + "\n" + json.dumps(second) + "\n")
    rows, excluded = load_checked(tmp_path)
    assert len(rows) == 1 and len(rows[0]["evidence"]) == 1
    assert rows[0]["source_pages"] == ["Other", "Page"]
    assert excluded["duplicate_evidence_variants_collapsed"] == 1


def test_partitions_share_no_claim_page_or_subject():
    rows = [
        {
            "id": str(i),
            "question": f"Claim {i}",
            "expected": ["YES", "NO", "MAYBE"][i % 3],
            "source_pages": [f"Page{i // 2}"],
            "subject": f"Subject{i // 2}",
        }
        for i in range(100)
    ]
    splits, _ = partition(
        rows, targets={"development": 20, "validation": 12, "calibration": 12, "evaluation": 12}
    )
    seen_ids, seen_pages, seen_subjects = set(), set(), set()
    for values in splits.values():
        ids = {r["id"] for r in values}
        pages = {p for r in values for p in r["source_pages"]}
        subjects = {r["subject"] for r in values}
        assert not ids & seen_ids and not pages & seen_pages and not subjects & seen_subjects
        seen_ids |= ids
        seen_pages |= pages
        seen_subjects |= subjects


def test_paired_metrics_reject_duplicate_ids():
    row = {"id": "a", "group_id": "g", "expected": "YES", "decision": "YES", "latency_seconds": 1}
    with pytest.raises(ValueError):
        paired_comparison([row, row], [row, row], replicates=5)
    result = summary([row])
    assert result["all_in_cost_usd"] is None and result["cost_per_correct_definitive_usd"] is None


@pytest.mark.parametrize("method", ["temperature", "logistic", "isotonic"])
def test_fit_serializable_mapping(method):
    rows = [
        {"candidate": "YES", "expected": "YES" if i % 3 else "NO", "raw_score": 0.6 + i * 0.01}
        for i in range(20)
    ]
    mapping = fit(method, rows)
    restored = json.loads(json.dumps(mapping))
    assert 0 <= predict(restored, 0.8) <= 1


def test_population_gate_and_exact_reuse():
    class Model:
        config: ClassVar[dict[str, str]] = {"revision": "fixed"}
        calls = 0

        def predict(self, question, evidence):
            self.calls += 1
            return {"candidate": "YES", "raw_score": 0.9, "truncated": False}

    artifact = {
        "model_config": Model.config,
        "population": "fixed-corpus",
        "mapping": {"method": "isotonic", "x": [0, 1], "y": [0, 1]},
        "certificate": {"qualified": True, "row_risk_upper_bootstrap": 0.1},
    }
    model = Model()
    pipeline = ResearchPipeline(model, artifact)
    evidence = [{"url": "https://example.org", "text": "A fact."}]
    assert (
        pipeline.verify_evidence("Claim", evidence, population="fixed-corpus")["decision"] == "YES"
    )
    assert pipeline.verify_evidence("Claim", evidence, population="fixed-corpus")["cache_hit"]
    pipeline.verify_evidence("Not claim", evidence, population="fixed-corpus")
    pipeline.verify_evidence(
        "Claim", [{**evidence[0], "text": "Changed."}], population="fixed-corpus"
    )
    assert model.calls == 3
    with pytest.raises(ValueError):
        pipeline.verify_evidence("Claim", evidence, population="new-world")
    artifact["certificate"]["qualified"] = False
    assert apply({"candidate": "YES", "raw_score": 0.99}, artifact)["decision"] == "MAYBE"


def test_head_json_no_pickle():
    checkpoint = {
        "scaler_mean": [0] * 5,
        "scaler_scale": [1] * 5,
        "weights": [[0] * 5] * 3,
        "biases": [0, 2, 0],
        "classes": [0, 1, 2],
    }
    result = infer_head(
        {"logits": [1, 0, 0], "token_count": 10}, checkpoint, evidence_group_count=1
    )
    assert result["candidate"] == "YES"
    assert infer_head({"logits": None}, checkpoint, evidence_group_count=0)["logits"] is None


def test_spend_persists_and_no_unapproved_calls(tmp_path):
    db = tmp_path / "spend.sqlite"
    ledger = SpendLedger(db, authorized_usd=0.03)
    ledger.reserve(0.02, "call")
    ledger.close()
    ledger = SpendLedger(db, authorized_usd=0.03)
    with pytest.raises(PermissionError):
        ledger.reserve(0.02, "call")
    ledger.close()
    with pytest.raises(ValueError):
        SpendLedger(tmp_path / "nan.sqlite", authorized_usd=float("nan"))


def fake_client():
    class Models:
        calls = 0

        def count_tokens(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(total_tokens=100)

        def generate_content(self, **kwargs):
            self.calls += 1
            assert kwargs["config"].thinking_config.thinking_level == "LOW"
            return SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=100, candidates_token_count=10, thoughts_token_count=20
                ),
                candidates=[],
                text='{"decision":"MAYBE","citations":[]}',
                model_version="version-fixed",
                model_dump=lambda **kwargs: {"modelVersion": "version-fixed"},
            )

    return SimpleNamespace(models=Models())


def test_vertex_requires_fresh_spending_authorization(tmp_path):
    client = fake_client()
    ledger = SpendLedger(tmp_path / "zero.sqlite")
    model = VertexGemini(VertexConfig(project="test"), ledger, client=client)
    with pytest.raises(PermissionError):
        model.answer("Question", [])
    assert client.models.calls == 0
    ledger.close()


def test_vertex_usage_keeps_thinking_and_unknown_invoice(tmp_path):
    client = fake_client()
    ledger = SpendLedger(tmp_path / "approved.sqlite", authorized_usd=1)
    model = VertexGemini(VertexConfig(project="test"), ledger, client=client)
    result = model.answer("Question", [])
    assert result["reasoning_tokens"] == 20 and result["output_tokens"] == 10
    assert result["token_cost_at_list_price_usd"] == pytest.approx((100 * 0.75 + 30 * 3.75) / 1e6)
    assert result["actual_invoiced_cost_usd"] is None and result["all_in_cost_usd"] is None
    with pytest.raises(PermissionError):
        model.answer("Question", grounded=True)
    ledger.close()


def test_gemini_settings_reject_unpriced_substitution_and_minimal():
    with pytest.raises(ValueError):
        VertexConfig(project="test", thinking_level="MINIMAL").validate()
    with pytest.raises(ValueError):
        VertexConfig(project="test", model="other").validate()


def test_priced_flash_substitutions():
    VertexConfig(project="test", model="gemini-3.7-flash").validate()
    VertexConfig(
        project="test",
        model="gemini-2.5-flash",
        thinking_budget=1024,
        input_usd_per_million=0.3,
        output_usd_per_million=2.5,
    ).validate()
    with pytest.raises(ValueError):
        VertexConfig(project="test", model="gemini-2.5-flash").validate()


def test_oracle_first_router_defers_uncertain_answers():
    from houndoracle.experiments.router import OracleFirstRouter

    class Model:
        artifact: ClassVar[dict[str, str]] = {"population": "frozen"}

        def verify_evidence(self, *args, **kwargs):
            return {"decision": "YES", "confidence": 0.82}

    class Gemini:
        calls = 0

        def answer(self, *args):
            self.calls += 1
            return {"answer": {"decision": "NO"}, "token_cost_at_list_price_usd": 0.001}

    gemini = Gemini()
    assert OracleFirstRouter(Model(), gemini).answer("Claim", [])["source"] == "oracle"
    assert gemini.calls == 0
    assert (
        OracleFirstRouter(Model(), gemini, minimum_confidence=0.85).answer("Claim", [])["decision"]
        == "NO"
    )
    assert gemini.calls == 1
    assert (
        OracleFirstRouter(Model(), gemini, bypass=True).answer("Claim", [])["source"]
        == "gemini_fallback"
    )
    assert gemini.calls == 2


@pytest.mark.parametrize(
    "indices,valid", [([0], True), ([], False), ([True], False), ([9], False), ([0, 0], False)]
)
def test_compact_citations_are_reconstructed_and_validated(tmp_path, indices, valid):
    client = fake_client()
    original = client.models.generate_content

    def generate(**kwargs):
        assert kwargs["config"].response_schema is not None
        response = original(**kwargs)
        response.text = json.dumps({"decision": "YES", "evidence_indices": indices})
        return response

    client.models.generate_content = generate
    ledger = SpendLedger(tmp_path / "compact.sqlite", authorized_usd=1)
    model = VertexGemini(
        VertexConfig(project="test", response_format="evidence_indices"), ledger, client=client
    )
    evidence = [{"url": "https://example.org", "text": "An exact original passage."}]
    result = model.answer("Question", evidence)
    assert result["output_valid"] is valid
    if valid:
        assert result["answer"]["citations"] == [
            {"url": evidence[0]["url"], "quote": evidence[0]["text"]}
        ]
    with pytest.raises(ValueError):
        model.answer("Question")
    ledger.close()
