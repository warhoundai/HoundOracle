import pytest

from houndoracle.experiments.negation import combine, negated_view


def test_unparsed_question_cannot_be_silently_negated():
    with pytest.raises(ValueError):
        negated_view("Is it false that X is not true?")
    text = "It is not true that X was released in2024."
    assert text in negated_view(text, declared_proposition=True)


def test_negation_swaps_yes_no_and_leaves_unknown_without_false_confidence():
    result = combine([-3, 5, 0], [5, -3, 0])
    assert result["candidate"] == "YES" and result["views_agree"]
    assert result["calibrated_correctness"] is None and not result["production_qualified"]
    assert combine([0, 0, 5], [0, 0, 5])["candidate"] == "MAYBE"
    assert not combine([-3, 5, 0], [-3, 5, 0])["views_agree"]


def test_bad_logits_do_not_manufacture_a_verdict():
    with pytest.raises(ValueError):
        combine([float("nan"), 1, 0], [0, 1, 0])
    with pytest.raises(ValueError):
        combine([1, 0], [0, 1, 0])
