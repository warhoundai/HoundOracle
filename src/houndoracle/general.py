"""Bounded general-proposition pipeline. Raw candidates never bypass calibration.

Acquisition qualification is an integrity/transport fact, not human review or
proof of factual truth. Origin groups are dependency hints, not independence.
"""

import hashlib
import heapq
import json
import math
import re
import time
from urllib.parse import urlsplit

from .budget import Budget, BudgetExceeded
from .compiler import cache_key
from .evaluators import LogisticAggregator
from .models import Decision, DocumentInput, EvaluationTrace
from .search import source_url
from .warhound import challenge_detected
from .web import acquire_source


class GroupedAggregator:
    def __init__(self, weights=(1.0, -1.0, 0.0), bias=0.0):
        self.model = LogisticAggregator(weights, bias)

    @property
    def version(self):
        return hashlib.sha256(("general-grouped-v2:" + self.model.version).encode()).hexdigest()

    def combine(self, observations):
        # Each origin/content/upstream group contributes at most one sign.
        groups = {}
        for row in observations:
            if (
                row["candidate"] in ["YES", "NO"]
                and not row["truncated"]
                and row["raw_score"] >= 0.5
            ):
                groups.setdefault(row["group"], set()).add(row["candidate"])
        signs = set().union(*groups.values()) if groups else set()

        def strongest(sign):
            scores = [
                r["raw_score"]
                for r in observations
                if r["candidate"] == sign and not r["truncated"] and r["raw_score"] >= 0.5
            ]
            p = min(max(max(scores, default=0.5), 0.5), 1 - 1e-7)
            return math.log(p / (1 - p))

        # Max pooling retains uncertainty without treating website count as
        # independent likelihood multiplication. This is still an uncalibrated score.
        vector = [strongest("YES"), strongest("NO"), float(len(signs) > 1)]
        z = self.model.bias + sum(w * x for w, x in zip(self.model.weights, vector))
        score = 1 / (1 + math.exp(-max(-50, min(50, z))))
        return {
            "candidate": next(iter(signs)) if len(signs) == 1 else "MAYBE",
            "raw_score": max(score, 1 - score),
            "conflict": len(signs) > 1,
            "groups": len(groups),
        }


def acquired_source(url, budget):
    return acquire_source(url, budget)


def dependency_groups(documents):
    parents = {}

    def root(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = root(parents[x])
        return parents[x]

    def union(a, b):
        parents[root(a)] = root(b)

    for r in documents:
        d = json.loads(r["data"])
        host = urlsplit(d["url"]).hostname or "unknown"
        # Do not count subdomains as independently corroborating publishers.
        origin = ".".join(host.split(".")[-2:]) if "." in host else host
        union(r["id"], "origin:" + origin)
        union(r["id"], "hash:" + r["hash"])
        for upstream in d.get("upstream_ids", []):
            union(r["id"], upstream)
    return {r["id"]: root(r["id"]) for r in documents}


class GeneralVerifier:
    def __init__(
        self,
        evaluator,
        *,
        aggregator=None,
        source_acquirer=acquired_source,
        acquirer_version="bounded-http-v1",
    ):
        self.evaluator = evaluator
        self.aggregator = aggregator or GroupedAggregator()
        self.source_acquirer = source_acquirer
        self.acquirer_version = acquirer_version

    @property
    def version(self):
        return hashlib.sha256(
            json.dumps(
                [
                    "general-pipeline-v4",
                    "lexical-window-ranking-round-robin-mandatory-counter-v1",
                    48,
                    self.evaluator.version,
                    self.aggregator.version,
                    self.acquirer_version,
                    800,
                    100,
                    16,
                    4,
                    12,
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def _evaluate(self, oracle, claim, budget, trace, memo, required_ids):
        leaf = claim
        negative = False
        while leaf.kind == "not":
            negative = not negative
            leaf = leaf.children[0]
        if leaf.kind != "prose":
            return None, [], False
        all_docs = oracle.store.documents()
        if len(all_docs) > 1000 or sum(len(r["data"]) for r in all_docs) > 5_000_000:
            raise ValueError("CORPUS_LIMIT")
        if len(all_docs) > 10:
            hits = oracle.store.search(leaf.proposition, 10)
            wanted = {h["id"]: i for i, h in enumerate(hits)}
            documents = sorted(
                [r for r in all_docs if r["id"] in wanted], key=lambda r: wanted[r["id"]]
            )
        else:
            documents = all_docs
        mandatory = [r for r in all_docs if r["id"] in required_ids]
        mandatory_ids = {r["id"] for r in mandatory}
        documents = (mandatory + [r for r in documents if r["id"] not in mandatory_ids])[:10]
        groups = dependency_groups(documents)
        observations = []
        citations = []
        qualified = 0
        windows = 0
        plans = []
        terms = set(re.findall(r"\w+", leaf.proposition.casefold())) - {
            "is",
            "was",
            "the",
            "a",
            "an",
            "it",
            "that",
            "in",
            "on",
            "of",
        }
        for r in documents:
            budget.check_time()
            d = json.loads(r["data"])
            if hashlib.sha256(d["text"].encode()).hexdigest() != r["hash"]:
                continue
            machine = (
                d.get("validation_status") == "acquired"
                and d.get("validation_method") == "bounded-source-acquisition-v1"
                and d.get("validation_content_hash") == r["hash"]
                and d.get("retrieval_kind") == "source_document"
            )
            if d.get("validation_status") == "revoked" or not (d["reviewed"] or machine):
                continue
            qualified += 1
            starts = []
            for start in range(0, len(d["text"]), 700):
                end = min(start + 800, len(d["text"]))
                lexical = len(terms & set(re.findall(r"\w+", d["text"][start:end].casefold())))
                starts.append((-lexical, start, end))
                if end == len(d["text"]):
                    break
            plans.append((r, d, sorted(starts)))
        evaluated_ids = set()
        for rank in range(16):
            for r, d, starts in plans:
                if rank >= len(starts) or windows >= 16:
                    continue
                _, start, end = starts[rank]
                passage = d["text"][start:end]
                identity = (r["id"], start, end)
                if identity not in memo:
                    if len(memo) >= 48:
                        continue
                    memo[identity] = self.evaluator.predict(leaf.proposition, passage)
                result = memo[identity]
                budget.check_time()
                windows += 1
                if (
                    result.get("candidate") not in ["YES", "NO", "MAYBE"]
                    or type(result.get("truncated")) is not bool
                    or type(result.get("raw_score")) not in (int, float)
                    or not math.isfinite(result["raw_score"])
                    or not 0 <= result["raw_score"] <= 1
                ):
                    raise ValueError("INVALID_SEMANTIC_OUTPUT")
                observations.append(result | {"group": groups[r["id"]]})
                from .models import Citation

                citations.append(
                    Citation(
                        document_id=r["id"],
                        url=d["url"],
                        content_hash=r["hash"],
                        start=start,
                        end=end,
                        text=passage,
                        observed_at=d["observed_at"],
                        expires_at=d["expires_at"],
                        group=groups[r["id"]],
                    )
                )
                if not result["truncated"]:
                    evaluated_ids.add(r["id"])
        combined = self.aggregator.combine(observations)
        combined["evaluated_document_ids"] = evaluated_ids
        if negative and combined["candidate"] in ["YES", "NO"]:
            combined["candidate"] = "NO" if combined["candidate"] == "YES" else "YES"
        trace.candidate_decision = Decision(combined["candidate"])
        trace.raw_score = combined["raw_score"]
        trace.contradiction_detected = combined["conflict"]
        trace.evidence_groups = combined["groups"]
        trace.evaluated_passages = windows
        trace.qualified_documents = qualified
        trace.documents_considered = len(documents)
        trace.evidence_sufficient = combined["candidate"] in ["YES", "NO"] and bool(citations)
        return combined, citations, trace.evidence_sufficient

    def request_policy_id(self, oracle, request):
        regime = [
            self.version,
            oracle.policy_id,
            request.allow_network,
            min(request.max_requests, 16),
            request.max_seconds,
            request.max_cost_usd,
        ]
        return hashlib.sha256(json.dumps(regime, sort_keys=True).encode()).hexdigest()

    def verify(self, oracle, request, claim, *, control=None):
        started = time.monotonic()
        policy_id = self.request_policy_id(oracle, request)
        initial_version = self.version

        def build_result(*args, **kwargs):
            result = oracle._result(*args, **kwargs)
            result.versions["policy"] = policy_id
            return result

        budget = Budget(min(request.max_requests, 16), request.max_seconds, request.max_cost_usd)
        trace = EvaluationTrace()
        timings = {"evaluation": 0.0, "retrieval": 0.0, "acquisition": 0.0}
        key = cache_key(claim, request.min_confidence, oracle.version + ":general:" + policy_id)
        cancellation = control() if control else None
        if cancellation:
            return build_result(request, claim, cancellation, started, budget)
        try:
            budget.check_time()
        except BudgetExceeded as exc:
            return build_result(request, claim, str(exc), started, budget)
        cached = oracle.store.cached(key)
        if cached and oracle.calibration and oracle.calibration.expires_at > time.time():
            cancellation = control() if control else None
            if cancellation:
                return build_result(request, claim, cancellation, started, budget)
            try:
                budget.check_time()
            except BudgetExceeded as exc:
                return build_result(request, claim, str(exc), started, budget)
            with oracle.store.lock:
                if not all(
                    oracle.store._fresh(c.document_id, time.time()) for c in cached.citations
                ):
                    return build_result(
                        request, claim, "EVIDENCE_CHANGED_DURING_EVALUATION", started, budget
                    )
            cached.question = request.question
            cached.claim = claim
            cached.cache_hit = True
            if cached.diagnostics:
                cached.diagnostics.stage_seconds = {"cache_lookup": time.monotonic() - started}
                cached.diagnostics.evaluated_passages = 0
                cached.diagnostics.search_actions = 0
                cached.diagnostics.retrieved_documents = 0
            cached.usage = {
                "requests": 0,
                "reserved_cost_usd": 0,
                "actual_cost_usd": None,
                "latency_seconds": time.monotonic() - started,
            }
            return cached
        actions = []
        seen_urls = set()
        seen_queries = set()
        fetches = 0
        citations = []
        leaf = claim
        while leaf.kind == "not":
            leaf = leaf.children[0]
        if leaf.kind != "prose":
            return build_result(request, claim, "UNSUPPORTED_COMPOSITE", started, budget)
        q = leaf.proposition
        for i, query in enumerate(
            [q, q + " primary source", q + " contradictory evidence", q + " official record"]
        ):
            heapq.heappush(actions, (i * 2, i, "search", query))
        failure = None
        contradiction_search_succeeded = False
        contradiction_pending = set()
        fetched_urls = set()
        contradiction_urls = set()
        acquired_ids = {}
        memo = {}
        while True:
            try:
                budget.check_time()
                if self.version != initial_version:
                    failure = "POLICY_CHANGED_DURING_EVALUATION"
                    break
                cancellation = control() if control else None
                if cancellation:
                    failure = cancellation
                    break
                step = time.perf_counter()
                required_ids = {
                    acquired_ids[url] for url in contradiction_urls if url in acquired_ids
                }
                combined, citations, sufficient = self._evaluate(
                    oracle, claim, budget, trace, memo, required_ids
                )
                timings["evaluation"] += time.perf_counter() - step
                if sufficient and oracle.calibration is not None:
                    confidence = oracle.calibration.confidence(
                        combined["raw_score"], claim.kinds(), policy_id
                    )
                    # Online acceptance includes a contradiction-directed search action.
                    if (
                        confidence is not None
                        and confidence >= request.min_confidence
                        and (
                            oracle.local_only
                            or not request.allow_network
                            or (
                                contradiction_search_succeeded
                                and not contradiction_pending
                                and required_ids <= set(combined["evaluated_document_ids"])
                            )
                        )
                    ):
                        if self.version != initial_version:
                            failure = "POLICY_CHANGED_DURING_EVALUATION"
                            break
                        trace.stage_seconds = timings
                        result = build_result(
                            request,
                            claim,
                            "SUFFICIENT_EVIDENCE",
                            started,
                            budget,
                            Decision(combined["candidate"]),
                            confidence,
                            citations,
                            diagnostics=trace,
                        )
                        budget.check_time()
                        with oracle.store.lock:
                            if not all(
                                oracle.store._fresh(c.document_id, time.time()) for c in citations
                            ):
                                failure = "EVIDENCE_CHANGED_DURING_EVALUATION"
                                break
                            return oracle.store.save(key, result)
                if oracle.local_only or not request.allow_network:
                    break
                if not actions:
                    break
                _, seq, kind, value = heapq.heappop(actions)
                if kind == "search":
                    if oracle.provider is None:
                        failure = "NO_SEARCH_PROVIDER"
                        break
                    normalized = value.casefold()
                    if normalized in seen_queries:
                        continue
                    seen_queries.add(normalized)
                    trace.search_actions += 1
                    step = time.perf_counter()
                    try:
                        hits = oracle.provider.retrieve(value, budget)
                    except BudgetExceeded:
                        raise
                    except Exception:  # noqa: BLE001 -- never leak provider keys or query-bearing exceptions
                        trace.source_failures.append("SEARCH_FAILED")
                        continue
                    timings["retrieval"] += time.perf_counter() - step
                    budget.check_time()
                    if seq == 2:
                        contradiction_search_succeeded = True
                    for index, hit in enumerate(hits[:3]):
                        url = source_url(hit.get("url"))
                        if seq == 2 and url:
                            contradiction_urls.add(url)
                            if url not in fetched_urls:
                                contradiction_pending.add(url)
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            heapq.heappush(
                                actions, (seq * 2 + 1, 100 + seq * 12 + index, "fetch", url)
                            )
                else:
                    if fetches >= 12:
                        continue
                    fetches += 1
                    before = budget.requests
                    step = time.perf_counter()
                    try:
                        hit = self.source_acquirer(value, budget)
                        if budget.requests <= before:
                            raise ValueError("ACQUIRER_DID_NOT_RESERVE")
                        budget.check_time()
                        url = source_url(hit.get("url"))
                        text = hit.get("text")
                        if (
                            not url
                            or urlsplit(url).hostname != urlsplit(value).hostname
                            or not isinstance(text, str)
                            or not 1 <= len(text) <= 1_000_000
                            or challenge_detected(text)
                        ):
                            raise ValueError("INVALID_SOURCE")
                        now = time.time()
                        document_id = oracle.store.ingest(
                            DocumentInput(
                                url=url,
                                text=text,
                                origin=urlsplit(url).hostname,
                                observed_at=now,
                                expires_at=now + 3600,
                                retrieved_by=hit.get("retrieval_provider", "source-acquirer"),
                                retrieval_kind="source_document",
                                validation_status="acquired",
                                validation_method="bounded-source-acquisition-v1",
                                validation_content_hash=hashlib.sha256(text.encode()).hexdigest(),
                                reviewed=False,
                                primary=False,
                            )
                        )
                        trace.retrieved_documents += 1
                        fetched_urls.add(value)
                        acquired_ids[value] = document_id
                        contradiction_pending.discard(value)
                    except BudgetExceeded:
                        raise
                    except Exception:  # noqa: BLE001 -- inaccessible documents never become false claims
                        trace.source_failures.append("SOURCE_ACQUISITION_FAILED")
                    timings["acquisition"] += time.perf_counter() - step
            except BudgetExceeded as exc:
                failure = str(exc)
                break
            except Exception:  # noqa: BLE001 -- isolate malformed evidence/model failures
                failure = "SEMANTIC_EVALUATION_FAILED"
                break
        trace.stage_seconds = timings
        reason = (
            failure
            or ("RETRIEVAL_FAILED" if trace.source_failures else None)
            or (
                "UNCALIBRATED"
                if trace.evidence_sufficient
                else "CONTRADICTORY_EVIDENCE"
                if trace.contradiction_detected
                else "INSUFFICIENT_EVIDENCE"
            )
        )
        return build_result(
            request, claim, reason, started, budget, citations=citations, diagnostics=trace
        )
