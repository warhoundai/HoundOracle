"""Bounded best-first verification states; no chess-style minimax assumptions."""

import hashlib
import heapq
import json
import time

from .budget import Budget, BudgetExceeded
from .compiler import VERSION as COMPILER_VERSION
from .compiler import cache_key, compile_question, equivalent, invalid_question
from .evaluators import LogisticAggregator, SemanticCompiler, atom_key, symbolic
from .models import Claim, Decision, DocumentInput, EvaluationTrace, Verification, VerifyRequest
from .store import Store


class Oracle:
    def __init__(
        self,
        store=None,
        *,
        calibration=None,
        provider=None,
        aggregator=None,
        local_only=True,
        general_verifier=None,
    ):
        self.store = store if store is not None else Store()
        self.local_only = local_only
        self.general_verifier = general_verifier
        if local_only and provider is not None:
            raise ValueError("Self-hosted offline serving cannot use an external search provider")
        self.calibration, self.provider = calibration, provider
        self.compiler = SemanticCompiler()
        self.aggregator = aggregator or LogisticAggregator()

    @property
    def policy_id(self):
        return hashlib.sha256(
            json.dumps(
                [
                    COMPILER_VERSION,
                    self.compiler.version,
                    self.aggregator.version,
                    "primary-reviewed-kleene-1",
                    "fts-provenance-fresh-before-limit-v2",
                    "all-atoms-min-score-1",
                    *(
                        [self.general_verifier.version, self.local_only]
                        if self.general_verifier is not None
                        else []
                    ),
                ]
            ).encode()
        ).hexdigest()

    @property
    def version(self):
        return (
            self.policy_id
            + ":"
            + (self.calibration.version if self.calibration else "uncalibrated")
        )

    def search(self, query, limit=10):
        """Local discovery only. These results are not verification decisions."""
        return self.store.search(query, limit)

    def web_search(
        self,
        query,
        limit=10,
        *,
        allow_network=False,
        max_requests=4,
        max_seconds=20,
        max_cost_usd=0,
        prefer_local=True,
    ):
        """Discover evidence with explicit privacy/cost permission, never a factual verdict."""
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 4000:
            raise ValueError("Query must contain1..4000characters")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("Web result limit must be1..10")
        if type(max_requests) is not int or not 1 <= max_requests <= 4:
            raise ValueError("Search request limit must be1..4")
        import math

        if (
            not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 120
            or not math.isfinite(max_cost_usd)
            or not 0 <= max_cost_usd <= 1
        ):
            raise ValueError("Invalid search time/cost budget")
        started = time.monotonic()
        if prefer_local:
            local = self.search(query, limit)
            if local:
                return {
                    "results": [
                        {
                            "document_id": r["id"],
                            "url": r["data"]["url"],
                            "text": r["data"]["text"],
                            "observed_at": r["data"]["observed_at"],
                            "expires_at": r["data"]["expires_at"],
                            "published_at": r["data"].get("published_at"),
                            "retrieval_provider": "local",
                            "original_retrieval_provider": r["data"].get("retrieved_by"),
                            "retrieval_kind": r["data"].get("retrieval_kind"),
                            "reviewed": r["data"]["reviewed"],
                            "primary": r["data"]["primary"],
                        }
                        for r in local
                    ],
                    "verified": False,
                    "source": "local",
                    "usage": {
                        "requests": 0,
                        "reserved_cost_usd": 0,
                        "latency_seconds": time.monotonic() - started,
                    },
                }
        if self.local_only:
            raise PermissionError("LOCAL_ONLY_NETWORK_DISABLED")
        if not allow_network:
            raise PermissionError("Live search requires allow_network=True")
        if self.provider is None:
            return {
                "results": [],
                "verified": False,
                "reason": "NO_SEARCH_PROVIDER",
                "usage": {"requests": 0, "reserved_cost_usd": 0},
            }
        budget = Budget(max_requests, max_seconds, max_cost_usd)
        from .search import normalize

        reason = None
        try:
            hits = self.provider.retrieve(query, budget)
            budget.check_time()
            hits = normalize(hits, getattr(self.provider, "name", "configured"), limit)
            for hit in hits:
                from urllib.parse import urlsplit

                hit["document_id"] = self.store.ingest(
                    DocumentInput(
                        url=hit["url"],
                        text=hit["text"],
                        origin=urlsplit(hit["url"]).hostname or "unknown",
                        published_at=hit.get("published_at"),
                        retrieved_by=hit.get("retrieval_provider"),
                        retrieval_kind=hit.get("retrieval_metadata", {}).get("kind"),
                        observed_at=hit["observed_at"],
                        expires_at=hit["observed_at"] + 3600,
                        reviewed=False,
                        primary=False,
                    )
                )
        except BudgetExceeded as exc:
            hits = []
            reason = str(exc)
        except Exception:  # noqa: BLE001 -- never expose provider credentials or query text
            hits = []
            reason = "RETRIEVAL_FAILED"
        return {
            "results": hits,
            "verified": False,
            "source": "external",
            "provider_status": getattr(self.provider, "last_status", {}),
            "reason": reason,
            "usage": {
                "requests": budget.requests,
                "reserved_cost_usd": budget.cost,
                "actual_cost_usd": None,
                "latency_seconds": time.monotonic() - started,
            },
        }

    def explain(self, result_id):
        return self.store.explain(result_id)

    def verify_claim(self, claim: Claim, **options):
        # Explicit structured interface avoids pretending an unparsed question is equivalent.
        question = "Verify structured proposition: " + claim.model_dump_json()
        return self.verify(question, claim=claim, **options)

    def verify(self, question, *, _control=None, **options):
        request = VerifyRequest(question=question, **options)
        start = time.monotonic()
        budget = Budget(request.max_requests, request.max_seconds, request.max_cost_usd)
        if self.general_verifier is not None:
            from .compiler import compile_general_question

            parsed = compile_general_question(question)
        else:
            parsed = compile_question(question)
        claim = request.claim or parsed
        if request.claim:
            exact_structured = False
            prefix = "Verify structured proposition: "
            if question.startswith(prefix):
                try:
                    exact_structured = equivalent(
                        Claim.model_validate_json(question[len(prefix) :]), request.claim
                    )
                except ValueError:
                    exact_structured = False
            if not exact_structured and (parsed is None or not equivalent(parsed, request.claim)):
                return self._result(request, None, "QUESTION_CLAIM_MISMATCH", start, budget)
        if claim is None and invalid_question(question):
            return self._result(request, None, "NOT_EVALUABLE", start, budget, Decision.INVALID)
        if claim is None:
            return self._result(request, None, "UNSUPPORTED_OR_AMBIGUOUS", start, budget)
        if "prose" in claim.kinds():
            if self.general_verifier is None:
                return self._result(
                    request, claim, "GENERAL_EVALUATOR_NOT_CONFIGURED", start, budget
                )
            return self.general_verifier.verify(self, request, claim, control=_control)
        key = cache_key(claim, request.min_confidence, self.version)
        if _control and _control():
            return self._result(request, claim, _control(), start, budget)
        with self.store.lock:
            cached = self.store.cached(key)
            if cached and self.calibration and self.calibration.expires_at > time.time():
                try:
                    budget.check_time()
                except BudgetExceeded as exc:
                    return self._result(request, claim, str(exc), start, budget)
                if cached.diagnostics:
                    cached.diagnostics.stage_seconds = {"cache_lookup": time.monotonic() - start}
                cached.question = question
                cached.claim = claim
                cached.cache_hit = True
                cached.usage = {
                    "requests": 0,
                    "reserved_cost_usd": 0,
                    "actual_cost_usd": None,
                    "latency_seconds": time.monotonic() - start,
                }
                return cached

        # Nodes are verification states: action priority, sequence, unresolved atom.
        frontier = []
        for index, atom in enumerate(claim.atoms()):
            heapq.heappush(frontier, (0, index, atom))
        seen = set()
        terminal = None
        while True:
            if _control and _control():
                terminal = _control()
            try:
                budget.check_time()
            except BudgetExceeded as exc:
                terminal = str(exc)
            with self.store.lock:
                size = self.store.db.execute(
                    "SELECT count(*),coalesce(sum(length(data)),0) FROM documents WHERE active=1"
                ).fetchone()
                if size[0] > 1000 or size[1] > 5_000_000:
                    return self._result(request, claim, "CORPUS_LIMIT", start, budget)
                docs = self.store.documents()
                if len(docs) > 1000:
                    return self._result(request, claim, "CORPUS_LIMIT", start, budget)
                evaluation_start = time.perf_counter()
                relations = self.compiler.evaluate(claim, docs)
                verdict = symbolic(claim, relations)
                # All atoms must be independently resolved before current composite policy qualifies.
                resolved = all(symbolic(a, relations) is not None for a in claim.atoms())
                scores = []
                for atom in claim.atoms():
                    local = [r for r in relations if r.atom == atom_key(atom)]
                    p = self.aggregator.probability(local)
                    scores.append(max(p, 1 - p))
                score = min(scores)
                diagnostics = EvaluationTrace(
                    candidate_decision=Decision.YES
                    if verdict is True and resolved
                    else Decision.NO
                    if verdict is False and resolved
                    else Decision.MAYBE,
                    raw_score=score,
                    evidence_sufficient=verdict is not None and resolved,
                    evidence_groups=len({r.citation.group for r in relations if r.reviewed}),
                    documents_considered=len(docs),
                    extracted_relations=len(relations),
                    stage_seconds={"evaluation": time.perf_counter() - evaluation_start},
                )
                confidence = (
                    self.calibration.confidence(score, claim.kinds(), self.policy_id)
                    if self.calibration and verdict is not None and resolved
                    else None
                )
                citations = list(
                    {
                        (r.citation.document_id, r.citation.start, r.citation.end): r.citation
                        for r in relations
                    }.values()
                )
                if _control and _control():
                    terminal = _control()
                # Final gate occurs after evaluation too: time exhaustion always abstains.
                try:
                    budget.check_time()
                except BudgetExceeded as exc:
                    terminal = str(exc)
                if (
                    terminal is None
                    and confidence is not None
                    and confidence >= request.min_confidence
                ):
                    result = self._result(
                        request,
                        claim,
                        "SUFFICIENT_EVIDENCE",
                        start,
                        budget,
                        Decision.YES if verdict else Decision.NO,
                        confidence,
                        citations,
                        diagnostics=diagnostics,
                    )
                    return self.store.save(key, result)
            if terminal:
                return self._result(
                    request,
                    claim,
                    terminal,
                    start,
                    budget,
                    citations=citations,
                    diagnostics=diagnostics,
                )
            if self.local_only or not request.allow_network or self.provider is None:
                reason = (
                    "UNCALIBRATED"
                    if verdict is not None and resolved and confidence is None
                    else "INSUFFICIENT_EVIDENCE"
                )
                return self._result(
                    request,
                    claim,
                    reason,
                    start,
                    budget,
                    citations=citations,
                    diagnostics=diagnostics,
                )
            if not frontier:
                return self._result(
                    request,
                    claim,
                    "SEARCH_EXHAUSTED",
                    start,
                    budget,
                    citations=citations,
                    diagnostics=diagnostics,
                )
            _, _, atom = heapq.heappop(frontier)
            query = (
                f'"{atom.entity}" release date primary record'
                if atom.kind == "release_year"
                else f'"{atom.entity}" {atom.property} {atom.unit} primary record'
            )
            if query in seen:
                continue
            seen.add(query)
            try:
                hits = self.provider.retrieve(query, budget)
                budget.check_time()
                for hit in hits[:10]:
                    text = hit.get("text") or hit.get("content")
                    if isinstance(text, dict):
                        text = text.get("text")
                    if isinstance(text, str) and text and isinstance(hit.get("url"), str):
                        from urllib.parse import urlsplit

                        self.store.ingest(
                            DocumentInput(
                                url=hit["url"],
                                text=text[:1_000_000],
                                origin=urlsplit(hit["url"]).hostname or "unknown",
                                observed_at=time.time(),
                                expires_at=time.time() + 3600,
                                retrieved_by=hit.get("retrieval_provider"),
                                retrieval_kind=hit.get("retrieval_metadata", {}).get("kind"),
                                reviewed=False,
                                primary=False,
                            )
                        )
            except BudgetExceeded as exc:
                terminal = str(exc)
            except Exception:  # noqa: BLE001 -- isolate untrusted provider failures without leaking secrets
                # Provider messages can contain credentials or private request data.
                terminal = "RETRIEVAL_FAILED"

    def _result(
        self,
        request,
        claim,
        reason,
        start,
        budget,
        decision=Decision.MAYBE,
        confidence=None,
        citations=None,
        diagnostics=None,
    ):
        return Verification(
            decision=decision,
            question=request.question,
            claim=claim,
            confidence=confidence,
            reason=reason,
            explanation=(
                "Grounded evidence establishes the asked proposition."
                if decision == Decision.YES
                else "Grounded evidence refutes the asked proposition."
                if decision == Decision.NO
                else "Input is not an evaluable proposition."
                if decision == Decision.INVALID
                else "A definitive evidence-backed decision is not justified: " + reason + "."
            ),
            citations=citations or [],
            diagnostics=diagnostics,
            versions={
                "policy": self.policy_id,
                "calibration": self.calibration.version if self.calibration else "none",
            },
            usage={
                "requests": budget.requests,
                "reserved_cost_usd": budget.cost,
                "actual_cost_usd": None,
                "latency_seconds": time.monotonic() - start,
            },
        )
