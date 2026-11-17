"""Oracle-first research workflow with an explicit Gemini fallback and no hidden calls."""

import copy
import hashlib
import json
import time
from collections import OrderedDict


class OracleFirstRouter:
    def __init__(
        self,
        oracle,
        gemini,
        *,
        minimum_confidence=0.8,
        bypass=False,
        complete_evidence=False,
        evaluation_cache_size=0,
        evaluation_cache_ttl=60.0,
        permission_scope=None,
        clock=time.monotonic,
        insufficiency_specialist=None,
        insufficiency_gate=None,
    ):
        if not 0.8 <= minimum_confidence <= 1:
            raise ValueError("Research routing cannot lower the .80 confidence floor")
        if (
            type(evaluation_cache_size) is not int
            or evaluation_cache_size < 0
            or not 0 < evaluation_cache_ttl <= 3600
        ):
            raise ValueError("Invalid evaluation cache bounds")
        if evaluation_cache_size and not permission_scope:
            raise ValueError("Evaluation reuse requires an explicit permission scope")
        if (insufficiency_specialist is None) != (insufficiency_gate is None):
            raise ValueError("A specialist and its qualified gate must be supplied together")
        self.insufficiency_specialist = insufficiency_specialist
        self.insufficiency_gate = insufficiency_gate
        self.complete_evidence = complete_evidence
        self.cache_size, self.cache_ttl = evaluation_cache_size, evaluation_cache_ttl
        self.permission_scope, self.clock = permission_scope, clock
        self.cache = OrderedDict()
        self.oracle, self.gemini = oracle, gemini
        self.minimum_confidence, self.bypass = minimum_confidence, bypass

    def answer(self, question, evidence, *, proposition_validated=False):
        start = time.perf_counter()
        oracle_result = None
        cache_hit = False
        if (
            self.complete_evidence
            and proposition_validated
            and not self.bypass
            and evidence == []
            and question.strip()
        ):
            return {
                "decision": "MAYBE",
                "source": "empty_permitted_corpus",
                "oracle": None,
                "gemini": None,
                "latency_seconds": time.perf_counter() - start,
                "oracle_seconds": 0,
                "token_cost_usd": 0,
                "model_calls": 0,
                "evaluation_cache_hit": False,
                "failure": None,
                "reason": "NO_PERMITTED_EVIDENCE",
                "confidence": None,
            }
        if not self.bypass:
            key = None
            if self.cache_size:
                key = hashlib.sha256(
                    json.dumps(
                        [
                            question,
                            evidence,
                            self.permission_scope,
                            self.oracle.artifact,
                            getattr(getattr(self.oracle, "model", None), "config", None),
                        ],
                        sort_keys=True,
                        allow_nan=False,
                    ).encode()
                ).hexdigest()
                cached = self.cache.pop(key, None)
                if cached and self.clock() < cached[0]:
                    oracle_result = copy.deepcopy(cached[1])
                    self.cache[key] = cached
                    cache_hit = True
            if oracle_result is None:
                oracle_result = self.oracle.verify_evidence(
                    question,
                    evidence,
                    population=self.oracle.artifact["population"],
                    use_cache=False,
                )
                if key is not None:
                    self.cache[key] = (self.clock() + self.cache_ttl, copy.deepcopy(oracle_result))
                    while len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
        oracle_seconds = time.perf_counter() - start
        if (
            oracle_result
            and oracle_result["decision"] in ("YES", "NO")
            and (oracle_result["confidence"] or 0) >= self.minimum_confidence
        ):
            return {
                "decision": oracle_result["decision"],
                "source": "oracle",
                "evaluation_cache_hit": cache_hit,
                "oracle": oracle_result,
                "gemini": None,
                "latency_seconds": time.perf_counter() - start,
                "oracle_seconds": oracle_seconds,
                "token_cost_usd": 0,
                "model_calls": 0,
                "failure": None,
            }
        if (
            not self.bypass
            and self.complete_evidence
            and proposition_validated
            and self.insufficiency_specialist is not None
            and self.insufficiency_gate.artifact["risk_certificate"].get("qualified") is True
        ):
            specialist = self.insufficiency_specialist
            selection = specialist.model.predict(question, evidence)
            if self.insufficiency_gate.allows(
                selection,
                model_config=specialist.model.config,
                verification_artifact=specialist.artifact,
            ):
                return {
                    "decision": "MAYBE",
                    "source": "calibrated_insufficiency_specialist",
                    "oracle": oracle_result,
                    "gemini": None,
                    "confidence": None,
                    "insufficiency_candidate": selection["candidate"],
                    "insufficiency_raw_score": selection.get("raw_score"),
                    "latency_seconds": time.perf_counter() - start,
                    "oracle_seconds": time.perf_counter() - start,
                    "evaluation_cache_hit": cache_hit,
                    "token_cost_usd": 0,
                    "model_calls": 0,
                    "failure": None,
                    "reason": "CALIBRATED_INSUFFICIENCY_ABSTENTION",
                }
        oracle_seconds = time.perf_counter() - start
        generated = self.gemini.answer(question, evidence)
        return {
            "decision": generated["answer"]["decision"],
            "source": "gemini_fallback",
            "evaluation_cache_hit": cache_hit,
            "oracle": oracle_result,
            "gemini": generated,
            "latency_seconds": time.perf_counter() - start,
            "oracle_seconds": oracle_seconds,
            "token_cost_usd": generated["token_cost_at_list_price_usd"],
            "model_calls": 1,
            "failure": None
            if generated.get("output_valid", True)
            else generated.get("failure_reason") or "INVALID_MODEL_OUTPUT",
        }
