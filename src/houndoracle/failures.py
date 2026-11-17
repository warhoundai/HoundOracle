"""Operational failure reasons cannot masquerade as correct epistemic abstentions."""

OPERATIONAL_FAILURE_REASONS = frozenset(
    {
        "TIME_EXHAUSTED",
        "BUDGET_EXHAUSTED",
        "RETRIEVAL_FAILED",
        "NO_SEARCH_PROVIDER",
        "SEMANTIC_EVALUATION_FAILED",
        "POLICY_CHANGED_DURING_EVALUATION",
        "EVIDENCE_CHANGED_DURING_EVALUATION",
        "CORPUS_LIMIT",
        "JOB_CANCELLED",
        "PROVIDER_DEADLINE_EXCEEDED",
        "SOURCE_ACQUISITION_FAILED",
    }
)


def operational_failure(reason):
    return reason if reason in OPERATIONAL_FAILURE_REASONS else None
