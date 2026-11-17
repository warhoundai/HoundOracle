import pytest

from houndoracle.experiments.draft_relation import decode, preserve_or_fallback


def test_missing_information_cannot_be_formatted_as_refutation():
    text = '<think>analysis</think>\n```json\n{"proposition":"P","result":"Missing"}\n```'
    result = preserve_or_fallback(text, "P", 1, {"candidate": "NO", "failure": None})
    assert result["candidate"] == "MAYBE" and result["failure"] is None
    assert not result["production_qualified"] and result["calibrated_correctness"] is None


def test_explicit_scope_and_public_polarity_are_preserved():
    text = "<think>analysis</think>\n**Answer:** NO"
    assert decode(text, "P", declared_proposition=True)["relation"] == "NO"
    with pytest.raises(ValueError):
        decode(text, "Is P false?")
    assert preserve_or_fallback(text, "P", 2, {})["failure"] == "UNQUALIFIED_DRAFT_SOURCE_SCOPE"


def test_conflict_or_changed_proposition_cannot_override_original_question():
    bad = '<think>analysis</think>\n{"claim":"different P","decision":"YES"}'
    assert preserve_or_fallback(bad, "P", 1, {})["failure"] == "DRAFT_PROPOSITION_MISMATCH"
    conflict = "<think>analysis</think>\nAnswer: YES\nVerdict: NO"
    assert (
        decode(conflict, "P", declared_proposition=True)["status"] == "CONFLICTING_DRAFT_VERDICTS"
    )


def test_ambiguous_not_supported_is_not_assumed_false():
    text = '<think>analysis</think>\n{"result":"not supported"}'
    result = preserve_or_fallback(text, "P", 1, {"candidate": "NO"})
    assert result["candidate"] == "MAYBE" and result["failure"] == "AMBIGUOUS_DRAFT_LABEL"
    assert decode("No final tag. Answer: YES", "P", declared_proposition=True)["relation"] is None
