"""Decode explicit draft conclusions without asking another model to change their meaning."""

import json
import re

RELATIONS = {
    "yes": "YES",
    "supports": "YES",
    "supported": "YES",
    "entailment": "YES",
    "entails": "YES",
    "no": "NO",
    "refutes": "NO",
    "refuted": "NO",
    "contradiction": "NO",
    "contradicts": "NO",
    "maybe": "MAYBE",
    "missing": "MAYBE",
    "insufficient": "MAYBE",
    "unknown": "MAYBE",
    "neutral": "MAYBE",
    "not enough info": "MAYBE",
    "insufficient evidence": "MAYBE",
    "invalid": "INVALID",
}


def decode(text, proposition, *, declared_proposition=False):
    if declared_proposition is not True:
        raise ValueError("Require a declared proposition, not an unparsed question")
    if not isinstance(text, str) or "</think>" not in text:
        return {"relation": None, "status": "NO_COMPLETED_DRAFT_FINALE"}
    final = text.rsplit("</think>", 1)[1]
    values = []
    unknown = False
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", final):
        try:
            obj, _ = decoder.raw_decode(final[match.start() :])
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        fields = [
            obj[k]
            for k in ["decision", "relation", "result", "label", "classification"]
            if k in obj
        ]
        if not fields:
            continue
        if any(k in obj and obj[k] != proposition for k in ["proposition", "claim", "question"]):
            return {"relation": None, "status": "DRAFT_PROPOSITION_MISMATCH"}
        for value in fields:
            normalized = " ".join(value.casefold().split()) if isinstance(value, str) else None
            if normalized not in RELATIONS:
                unknown = True
            else:
                values.append(RELATIONS[normalized])
    for line in final.splitlines():
        line = line.replace("**", "").strip()
        found = re.fullmatch(
            r"(?:final\s+)?(?:answer|verdict|classification|result)\s*:\s*(.+?)[.!]?\s*",
            line,
            re.IGNORECASE,
        )
        if found:
            label = " ".join(found[1].casefold().split())
            if label in RELATIONS:
                values.append(RELATIONS[label])
            else:
                unknown = True
    if len(set(values)) > 1:
        return {"relation": None, "status": "CONFLICTING_DRAFT_VERDICTS"}
    if unknown:
        return {"relation": None, "status": "AMBIGUOUS_DRAFT_LABEL"}
    if not values:
        return {"relation": None, "status": "NO_EXPLICIT_DRAFT_VERDICT"}
    return {"relation": values[0], "status": "EXPLICIT_DRAFT_RELATION"}


def preserve_or_fallback(text, proposition, evidence_count, formatted):
    verdict = decode(text, proposition, declared_proposition=True)
    status = verdict["status"]
    if status in [
        "DRAFT_PROPOSITION_MISMATCH",
        "CONFLICTING_DRAFT_VERDICTS",
        "AMBIGUOUS_DRAFT_LABEL",
    ]:
        return {
            "candidate": "MAYBE",
            "evidence_indices": [],
            "failure": status,
            "normalizer_status": status,
            "calibrated_correctness": None,
            "production_qualified": False,
        }
    if verdict["relation"] is None:
        return formatted | {"normalizer_status": status}
    if evidence_count != 1:
        return {
            "candidate": "MAYBE",
            "evidence_indices": [],
            "failure": "UNQUALIFIED_DRAFT_SOURCE_SCOPE",
            "normalizer_status": status,
            "calibrated_correctness": None,
            "production_qualified": False,
        }
    candidate = verdict["relation"]
    return {
        "candidate": candidate,
        "evidence_indices": [0] if candidate in ["YES", "NO"] else [],
        "failure": None,
        "normalizer_status": status,
        "citation_binding": "Only permitted passage; not an independent entailment audit",
        "calibrated_correctness": None,
        "production_qualified": False,
    }
