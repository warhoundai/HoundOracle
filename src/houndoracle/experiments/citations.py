"""Matched snapshot citation integrity; this does not prove factual entailment."""


def citations_match(answer, evidence):
    citations = answer.get("citations")
    if not isinstance(citations, list) or not citations:
        return False
    for citation in citations:
        if not isinstance(citation, dict):
            return False
        url = citation.get("url") or citation.get("URL")
        quote = citation.get("quote") or citation.get("text")
        if not isinstance(url, str) or not isinstance(quote, str) or not quote.strip():
            return False
        if not any(
            e.get("url") == url and isinstance(e.get("text"), str) and quote in e["text"]
            for e in evidence
        ):
            return False
    return True
