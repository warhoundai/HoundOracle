"""Apply supported completion-level stopping to a synchronous chat formatter."""


def chat_with_stopping(model, stopping_criteria, **kwargs):
    if not callable(stopping_criteria):
        raise TypeError("Require a callable stopping control")
    original = model.create_completion
    missing = object()
    saved = model.__dict__.get("create_completion", missing)
    used = [False]

    def controlled(*args, **options):
        used[0] = True
        prior = options.pop("stopping_criteria", None)
        if prior is not None:
            if not callable(prior):
                raise TypeError("Unsupported formatter stopping policy")

            def combined(ids, logits):
                return bool(prior(ids, logits) or stopping_criteria(ids, logits))

            options["stopping_criteria"] = combined
        else:
            options["stopping_criteria"] = stopping_criteria
        return original(*args, **options)

    model.create_completion = controlled
    try:
        result = model.create_chat_completion(**kwargs)
        if not used[0]:
            raise RuntimeError("Chat handler bypassed completion stopping control")
        return result
    finally:
        if saved is missing:
            del model.create_completion
        else:
            model.create_completion = saved
