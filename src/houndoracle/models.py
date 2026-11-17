"""Strict, versioned wire contracts. No model-generated executable expressions."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Decision(str, Enum):
    YES = "YES"
    NO = "NO"
    MAYBE = "MAYBE"
    INVALID = "INVALID"


class Claim(Strict):
    kind: Literal["release_year", "quantity", "prose", "not", "and", "or"]
    proposition: str | None = Field(None, min_length=1, max_length=4000)
    entity: str | None = Field(None, min_length=1, max_length=200)
    property: str | None = Field(None, min_length=1, max_length=100)
    operator: Literal["eq", "lt", "gt", "le", "ge"] = "eq"
    value: str | None = Field(None, max_length=100)
    unit: Literal["kg", "g", "m", "cm", "s", "min", "count"] | None = None
    children: list[Claim] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def structure(self):
        if self.kind == "prose":
            if (
                not self.proposition
                or not self.proposition.strip()
                or self.children
                or any(x is not None for x in (self.entity, self.property, self.value, self.unit))
                or self.operator != "eq"
            ):
                raise ValueError("Prose preserves one exact proposition without atomic fields")
            return self
        if self.proposition is not None:
            raise ValueError("Only prose claims carry a proposition")
        if self.kind in ("not", "and", "or"):
            if len(self.children) < (1 if self.kind == "not" else 2):
                raise ValueError("Boolean claim requires operands")
            if self.kind == "not" and len(self.children) != 1:
                raise ValueError("Negation has exactly one operand")
            if any(x is not None for x in (self.entity, self.property, self.value, self.unit)):
                raise ValueError("Boolean claims cannot carry atomic fields")
        else:
            if not self.entity or self.value is None or self.children:
                raise ValueError("Atomic claim requires entity/value and no children")
            if self.kind == "release_year":
                if not self.value.isdigit() or not 1000 <= int(self.value) <= 9999:
                    raise ValueError("Expected four-digit release year")
                if self.property is not None or self.unit is not None:
                    raise ValueError("Release claim cannot carry quantity fields")
            else:
                from decimal import Decimal, InvalidOperation

                try:
                    if (
                        not Decimal(self.value).is_finite()
                        or abs(Decimal(self.value).adjusted()) > 12
                        or len(Decimal(self.value).as_tuple().digits) > 24
                    ):
                        raise ValueError("Nonfinite quantity")
                except InvalidOperation as exc:
                    raise ValueError("Invalid quantity") from exc
                if not self.property or not self.unit:
                    raise ValueError("Quantity requires property and unit")
        return self

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        data = handler(self)
        if self.proposition is None:
            data.pop("proposition", None)
        return data

    def kinds(self):
        return {self.kind}.union(*(child.kinds() for child in self.children))

    def atoms(self):
        if self.children:
            return [a for c in self.children for a in c.atoms()]
        return [self]


class VerifyRequest(Strict):
    question: str = Field(min_length=1, max_length=4000)
    claim: Claim | None = None
    min_confidence: float = Field(0.8, ge=0.8, le=1)
    max_requests: int = Field(4, ge=0, le=20)
    max_seconds: float = Field(10, gt=0, le=120)
    max_cost_usd: float = Field(0, ge=0, le=100)
    allow_network: bool = False


class Citation(Strict):
    document_id: str
    url: str
    content_hash: str
    start: int
    end: int
    text: str
    observed_at: float
    expires_at: float
    group: str


class EvaluationTrace(Strict):
    """Research diagnostics only. candidate_decision is NOT an accepted answer."""

    candidate_decision: Decision = Decision.MAYBE
    raw_score: float | None = Field(None, ge=0, le=1)
    evidence_sufficient: bool = False
    evidence_groups: int = Field(0, ge=0)
    documents_considered: int = Field(0, ge=0)
    extracted_relations: int = Field(0, ge=0)
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    evaluated_passages: int = Field(0, ge=0)
    qualified_documents: int = Field(0, ge=0)
    retrieved_documents: int = Field(0, ge=0)
    search_actions: int = Field(0, ge=0)
    source_failures: list[str] = Field(default_factory=list)
    contradiction_detected: bool = False

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        data = handler(self)
        # Keep existing deterministic diagnostic payloads compatible with strict
        # older clients; populated general diagnostics are an opt-in extension.
        for key in [
            "evaluated_passages",
            "qualified_documents",
            "retrieved_documents",
            "search_actions",
            "source_failures",
            "contradiction_detected",
        ]:
            if not data.get(key):
                data.pop(key, None)
        return data


class Verification(Strict):
    schema_version: str = "1"
    decision: Decision
    question: str
    claim: Claim | None = None
    confidence: float | None = None
    reason: str
    explanation: str
    citations: list[Citation] = Field(default_factory=list)
    diagnostics: EvaluationTrace | None = None
    result_id: str | None = None
    cache_hit: bool = False
    versions: dict[str, str] = Field(default_factory=dict)
    usage: dict[str, float | int | None] = Field(default_factory=dict)


class DocumentInput(Strict):
    url: str = Field(min_length=1, max_length=2048)
    text: str = Field(min_length=1, max_length=1_000_000)
    origin: str = Field(min_length=1, max_length=200)
    observed_at: float = Field(ge=0)
    expires_at: float = Field(ge=0)
    published_at: float | None = Field(None, ge=0)
    retrieved_by: str | None = Field(None, max_length=100)
    retrieval_kind: Literal["search_snippet", "source_document"] | None = None
    primary: bool = False
    reviewed: bool = False
    validation_status: Literal["unvalidated", "acquired", "revoked"] = "unvalidated"
    validation_method: str | None = Field(None, max_length=100)
    validation_content_hash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    public: bool = False
    license: str | None = Field(None, max_length=200)
    upstream_ids: list[str] = Field(default_factory=list, max_length=100)
    etag: str | None = None
    last_modified: str | None = None

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        data = handler(self)
        if (
            self.validation_status == "unvalidated"
            and self.validation_method is None
            and self.validation_content_hash is None
        ):
            for key in ["validation_status", "validation_method", "validation_content_hash"]:
                data.pop(key, None)
        return data

    @model_validator(mode="after")
    def dates(self):
        if self.validation_status == "revoked" and self.reviewed:
            raise ValueError("Revoked evidence cannot retain human review approval")
        if self.validation_status == "acquired":
            import hashlib

            if (
                self.retrieval_kind != "source_document"
                or self.validation_method != "bounded-source-acquisition-v1"
                or self.validation_content_hash != hashlib.sha256(self.text.encode()).hexdigest()
            ):
                raise ValueError("Acquisition qualification must bind the source document content")
        if self.expires_at <= self.observed_at:
            raise ValueError("Expiry must follow observation")
        return self
