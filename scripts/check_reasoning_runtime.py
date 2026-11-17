"""Exercise the real chat formatter API without allocating or querying a model."""

from llama_cpp import StoppingCriteriaList
from llama_cpp.llama_chat_format import (
    ChatFormatterResponse,
    chat_formatter_to_chat_completion_handler,
)

from houndoracle.experiments.completion_control import chat_with_stopping


def check(prior):
    def formatter(messages, **kwargs):
        return ChatFormatterResponse(prompt="probe", stop=[], stopping_criteria=prior)

    handler = chat_formatter_to_chat_completion_handler(formatter)

    class Probe:
        def tokenize(self, *args, **kwargs):
            return [1]

        def create_completion(self, **kwargs):
            self.control = kwargs["stopping_criteria"]
            return {
                "id": "cmpl-test",
                "object": "text_completion",
                "created": 0,
                "model": "probe",
                "choices": [{"index": 0, "text": "{}", "finish_reason": "stop", "logprobs": None}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        def create_chat_completion(self, **kwargs):
            return handler(llama=self, **kwargs)

    model = Probe()
    result = chat_with_stopping(
        model,
        StoppingCriteriaList([lambda ids, scores: True]),
        messages=[{"role": "user", "content": "probe"}],
    )
    if (
        result["choices"][0]["message"]["content"] != "{}"
        or not model.control([], [])
        or "create_completion" in model.__dict__
    ):
        raise RuntimeError("Native formatter deadline integration failed")


def main():
    for prior in [
        None,
        StoppingCriteriaList([lambda ids, scores: False]),
        StoppingCriteriaList([lambda ids, scores: True]),
    ]:
        check(prior)
    print("Real chat formatter integration passed; no model loaded or queried.")


if __name__ == "__main__":
    main()
