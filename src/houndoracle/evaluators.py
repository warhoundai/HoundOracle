"""Two evaluator interfaces; deterministic compiler + grouped logistic baseline.

NLI is an optional experiment. Its scores cannot authorize production decisions.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal

from .models import Citation, Claim

VERSION = "evaluators-1"
FACT = re.compile(
    r"(?m)^(?P<entity>[A-Za-z0-9][A-Za-z0-9 ._+/-]{0,199}?) was released in (?P<year>\d{4})\.$"
)
QUANTITY = re.compile(
    r"(?m)^(?P<entity>[A-Za-z0-9][A-Za-z0-9 ._+/-]{0,199}?) has (?P<prop>[A-Za-z][A-Za-z ]{0,99}?) (?P<value>-?\d+(?:\.\d+)?) (?P<unit>kg|g|m|cm|s|min|count)\.$"
)
UNITS = {
    "kg": ("mass", Decimal(1)),
    "g": ("mass", Decimal("0.001")),
    "m": ("length", Decimal(1)),
    "cm": ("length", Decimal("0.01")),
    "s": ("time", Decimal(1)),
    "min": ("time", Decimal(60)),
    "count": ("count", Decimal(1)),
}


@dataclass
class Relation:
    atom: str
    supports: bool
    citation: Citation
    primary: bool
    reviewed: bool


def atom_key(claim):
    from .compiler import canonical

    return json.dumps(canonical(claim), sort_keys=True)


def compare(left, op, right):
    return {
        "eq": left == right,
        "lt": left < right,
        "gt": left > right,
        "le": left <= right,
        "ge": left >= right,
    }[op]


class SemanticCompiler:
    """Exact controlled-language extraction. Arbitrary prose remains ungrounded.

    A whole line must match; negated, quoted, qualified or injected text is not
    silently converted into a grounded assertion. Local curator approval is
    still required: syntactic extraction alone does not establish truth.
    """

    version = VERSION

    def evaluate(self, claim: Claim, documents):
        relations = []
        # Hashes and explicit common origins induce transitive dependency groups.
        parents = {}

        def root(x):
            parents.setdefault(x, x)
            if parents[x] != x:
                parents[x] = root(parents[x])
            return parents[x]

        def union(a, b):
            parents[root(a)] = root(b)

        for row in documents:
            d = json.loads(row["data"])
            union(row["id"], "hash:" + row["hash"])
            union(row["id"], "origin:" + d["origin"])
            for upstream in d["upstream_ids"]:
                union(row["id"], upstream)
        for row in documents:
            d = json.loads(row["data"])
            for atom in claim.atoms():
                matches = (
                    FACT.finditer(d["text"])
                    if atom.kind == "release_year"
                    else QUANTITY.finditer(d["text"])
                )
                for m in matches:
                    if m["entity"] != atom.entity:
                        continue
                    if atom.kind == "quantity":
                        if m["prop"] != atom.property or UNITS[m["unit"]][0] != UNITS[atom.unit][0]:
                            continue
                        left = Decimal(m["value"]) * UNITS[m["unit"]][1]
                        right = Decimal(atom.value) * UNITS[atom.unit][1]
                    else:
                        left, right = int(m["year"]), int(atom.value)
                    citation = Citation(
                        document_id=row["id"],
                        url=d["url"],
                        content_hash=row["hash"],
                        start=m.start(),
                        end=m.end(),
                        text=m.group(),
                        observed_at=d["observed_at"],
                        expires_at=d["expires_at"],
                        group=root(row["id"]),
                    )
                    relations.append(
                        Relation(
                            atom_key(atom),
                            compare(left, atom.operator, right),
                            citation,
                            d["primary"],
                            d["reviewed"],
                        )
                    )
        return relations


def symbolic(claim, relations):
    """Strong Kleene composition. Conflicts remain unknown, including under NOT."""
    if claim.kind == "not":
        child = symbolic(claim.children[0], relations)
        return None if child is None else not child
    if claim.kind in ("and", "or"):
        children = [symbolic(c, relations) for c in claim.children]
        if claim.kind == "and":
            return (
                False if False in children else (True if all(c is True for c in children) else None)
            )
        return True if True in children else (False if all(c is False for c in children) else None)
    eligible = [r for r in relations if r.atom == atom_key(claim) and r.reviewed]
    values = {r.supports for r in eligible}
    # Definitive atomic facts require an explicitly reviewed primary source.
    if len(values) == 1 and any(r.primary for r in eligible):
        return values.pop()
    return None


def features(relations):
    groups = {}
    for r in relations:
        if r.reviewed:
            groups.setdefault(r.citation.group, set()).add(r.supports)
    support = sum(v == {True} for v in groups.values())
    against = sum(v == {False} for v in groups.values())
    conflict = sum(len(v) == 2 for v in groups.values())
    return [math.log1p(support), math.log1p(against), math.log1p(conflict)]


class LogisticAggregator:
    """Grouped logistic baseline. Default weights are heuristics, not training results."""

    def __init__(self, weights=(1.0, -1.0, 0.0), bias=0.0):
        self.weights, self.bias = tuple(weights), bias
        if len(self.weights) != 3 or not all(math.isfinite(x) for x in (*self.weights, bias)):
            raise ValueError("Expected three finite weights and finite bias")

    @property
    def version(self):
        return hashlib.sha256(json.dumps([VERSION, self.weights, self.bias]).encode()).hexdigest()

    def probability(self, relations):
        score = self.bias + sum(w * x for w, x in zip(self.weights, features(relations)))
        return 1 / (1 + math.exp(-max(-50, min(50, score))))

    def fit(self, examples, *, steps=300, learning_rate=0.05, l2=0.01):
        if not examples:
            raise ValueError("No training examples")
        weights = list(self.weights)
        bias = self.bias
        for _ in range(steps):
            dw, db = [0.0] * 3, 0.0
            for vector, label in examples:
                if (
                    len(vector) != 3
                    or label not in (0, 1)
                    or not all(math.isfinite(v) for v in vector)
                ):
                    raise ValueError("Invalid training example")
                z = bias + sum(w * x for w, x in zip(weights, vector))
                delta = 1 / (1 + math.exp(-max(-50, min(50, z)))) - label
                db += delta
                for i in range(3):
                    dw[i] += delta * vector[i]
            bias -= learning_rate * db / len(examples)
            for i in range(3):
                weights[i] -= learning_rate * (dw[i] / len(examples) + l2 * weights[i])
        self.weights, self.bias = tuple(weights), bias
        return self


class LocalNLI:
    """Research-only pinned local model. No implicit downloads or remote code."""

    def __init__(self, model_path, revision):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not revision:
            raise ValueError("An immutable model revision is required")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        self.model.eval()
        self.revision = revision

    def evaluate(self, claim, passage):
        import torch

        inputs = self.tokenizer(
            passage, claim, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.inference_mode():
            probabilities = self.model(**inputs).logits.softmax(-1)[0].tolist()
        return {self.model.config.id2label[i]: p for i, p in enumerate(probabilities)}
