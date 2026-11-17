"""Conservative compiler: semantic similarity never authorizes answer reuse."""

import hashlib
import json
import re

from .models import Claim

VERSION = "compiler-1"
ENTITY = r"[A-Za-z0-9][A-Za-z0-9 ._+/-]{0,199}?"


def compile_question(question: str) -> Claim | None:
    text = question.strip().rstrip("?.").strip()
    if len(text) > 4000:
        return None
    for prefix, negate in (("Is it false that ", True), ("Is it true that ", False)):
        if text.lower().startswith(prefix.lower()):
            child = compile_question(text[len(prefix) :])
            return Claim(kind="not", children=[child]) if child and negate else child
    # Explicit negation is scoped over the entire release-year proposition.
    match = re.fullmatch(rf"(?:Was |was )?({ENTITY}) (?:was )?(not )?released in (\d{{4}})", text)
    if not match:
        return None
    entity, negative, year = match.groups()
    try:
        atom = Claim(kind="release_year", entity=entity.strip(), value=year)
    except ValueError:
        return None
    return Claim(kind="not", children=[atom]) if negative else atom


def canonical(claim: Claim) -> dict:
    if claim.kind == "not":
        child = claim.children[0]
        if child.kind == "not":
            return canonical(child.children[0])
        return {"kind": "not", "children": [canonical(child)]}
    if claim.kind in ("and", "or"):
        children = [canonical(c) for c in claim.children]
        return {
            "kind": claim.kind,
            "children": sorted(children, key=lambda x: json.dumps(x, sort_keys=True)),
        }
    result = claim.model_dump(exclude_none=True)
    # Preserve entity spelling and punctuation: aliases/case can identify different products.
    if claim.kind == "quantity":
        from decimal import Decimal

        result["value"] = str(Decimal(claim.value).normalize())
    return result


def equivalent(a: Claim, b: Claim) -> bool:
    return canonical(a) == canonical(b)


def cache_key(claim: Claim, confidence: float, policy: str) -> str:
    payload = [VERSION, canonical(claim), confidence, policy]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def invalid_question(question: str) -> bool:
    return not question.strip() or bool(
        re.match(
            r"^(?:write|draw|translate|summarize|generate|search for|tell me a joke)\b",
            question.strip(),
            re.IGNORECASE,
        )
    )


def compile_general_question(question: str, *, depth=0) -> Claim | None:
    """Opt-in exact-text fallback. No paraphrase or similarity authorizes equivalence."""
    if depth > 8 or not isinstance(question, str) or not 1 <= len(question) <= 4000:
        return None
    structured = compile_question(question)
    if structured is not None:
        return structured
    if invalid_question(question):
        return None
    for prefix, negate in (("Is it false that ", True), ("Is it true that ", False)):
        if question.startswith(prefix):
            child = compile_general_question(question[len(prefix) :], depth=depth + 1)
            if child is None:
                return None
            return Claim(kind="not", children=[child]) if negate else child
    return Claim(kind="prose", proposition=question)
