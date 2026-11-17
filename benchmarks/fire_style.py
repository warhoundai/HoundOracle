"""Independent FIRE-style control-loop approximation, not a paper reproduction.

Callers supply generation/search functions and a Budget. No vendor SDK, licensed
FIRE source or data is included. Equality replaces SBERT similarity (documented
non-comparability); model/provider pricing and traces must be recorded by callers.
"""

from houndoracle.budget import BudgetExceeded


def run(
    claim,
    generate,
    search,
    budget,
    *,
    generation_cost_upper,
    search_cost_upper,
    max_steps=5,
    evidence_required=False,
):
    evidence, queries, calls = [], set(), 0
    try:
        for step in range(max_steps + 1):
            forced = step == max_steps
            budget.reserve(generation_cost_upper)
            calls += 1
            action = generate(claim=claim, evidence=list(evidence), force_final=forced)
            budget.check_time()
            if set(action) == {"answer"} and action["answer"] in ("YES", "NO"):
                if evidence_required and not evidence:
                    return {"decision": "MAYBE", "reason": "NO_EVIDENCE", "llm_calls": calls}
                return {
                    "decision": action["answer"],
                    "evidence": evidence,
                    "llm_calls": calls,
                    "reason": "FORCED_BINARY" if forced else "MODEL_SELECTED",
                }
            if forced or set(action) != {"query"} or not isinstance(action["query"], str):
                return {"decision": "MAYBE", "reason": "INVALID_MODEL_OUTPUT", "llm_calls": calls}
            query = action["query"].strip()
            if not query or query in queries:
                return {"decision": "MAYBE", "reason": "REPEATED_QUERY", "llm_calls": calls}
            queries.add(query)
            budget.reserve(search_cost_upper)
            evidence.append(search(query))
            budget.check_time()
    except BudgetExceeded as exc:
        return {"decision": "MAYBE", "reason": str(exc), "llm_calls": calls}
