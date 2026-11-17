import pytest

from houndoracle.experiments.completion_control import chat_with_stopping


class Model:
    def create_completion(self, **kwargs):
        return kwargs

    def create_chat_completion(self, **kwargs):
        return self.create_completion(prompt="rendered", **kwargs)


def test_stopping_reaches_supported_api_and_override_is_removed():
    model = Model()
    stopping = lambda ids, logits: False
    result = chat_with_stopping(model, stopping, messages=[])
    assert result["stopping_criteria"] is stopping
    assert "create_completion" not in model.__dict__


def test_formatter_failure_restores_native_method():
    class Broken(Model):
        def create_chat_completion(self, **kwargs):
            raise ValueError("formatter failure")

    model = Broken()
    with pytest.raises(ValueError):
        chat_with_stopping(model, lambda ids, logits: False, messages=[])
    assert "create_completion" not in model.__dict__


def test_bypassing_controller_cannot_be_called_a_bounded_completion():
    class Bypass(Model):
        def create_chat_completion(self, **kwargs):
            return {}

    with pytest.raises(RuntimeError):
        chat_with_stopping(Bypass(), lambda ids, logits: False, messages=[])


@pytest.mark.parametrize("prior", [None, False, True])
def test_formatter_stop_rule_is_composed_without_replacing_it(prior):
    class Formatter(Model):
        def create_chat_completion(self, **kwargs):
            return self.create_completion(
                stopping_criteria=None if prior is None else lambda ids, logits: prior
            )

    for timeout in [False, True]:
        result = chat_with_stopping(Formatter(), lambda ids, logits, value=timeout: value)
        assert result["stopping_criteria"]([], []) == bool(prior or timeout)
