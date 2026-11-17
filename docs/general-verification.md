# Local general-proposition verification (research preview)

The deterministic engine remains the default. General verification is opt-in and
requires a locally installed three-label NLI checkpoint, its safetensors SHA-256,
and the semantic Python dependencies. The runtime loads local files only and
rejects remote model code. Acquire model files separately and review their license.

```python
from houndoracle.engine import Oracle
from houndoracle.general import GeneralVerifier
from houndoracle.local_semantics import LocalPairEvaluator
from houndoracle.store import Store

pair = LocalPairEvaluator(
    "/absolute/path/to/local/checkpoint",
    expected_weights_sha256="<verified 64-character checkpoint SHA-256>",
)
oracle = Oracle(
    Store("oracle.sqlite"),
    local_only=True,
    general_verifier=GeneralVerifier(pair),
)
result = oracle.verify("Is it true that the product was released in 2024?")
```

Without qualified evidence and policy calibration this returns MAYBE. A raw NLI
candidate is diagnostic information, not permission to answer definitively.
Ingest trusted local source snapshots through the existing document interface;
do not mark unreviewed passages as human reviewed.

The CLI accepts `--semantic-model` and `--semantic-weights-sha256` before its
subcommand. Optional adapters require both `--semantic-adapter` and
`--semantic-adapter-sha256`. `--web` enables the configured self-hosted SearXNG
adapter and existing WarHound acquisition integration. Source snippets alone are
insufficient. Searches, full-document acquisition, contradiction investigation,
request budgets and deadlines remain bounded. No paid model fallback is required.
An Oracle configured this way also serves the existing REST/MCP interfaces.

Exact prose identity governs answer reuse. Explicit outer true/false wrappers
compose polarity; arbitrary paraphrases, multi-hop Boolean decomposition and
entity/time interpretation are not solved by this compiler. Same-origin and
content grouping provide dependency hints, not proof of source independence.

Calibration binds model assets, aggregation, controller and request policy.
Research artifacts cannot authorize definitive serving decisions. The campaign47
artifact belongs to archived v3; active v4 requires fresh exact-policy validation
and calibration. Changing acquisition behavior requires an updated acquisition
version and requalification. The 80% confidence floor does not imply 95% accuracy.

Run `python scripts/qualification95.py status --help` for readiness inspection.
The two-track audit, validation and locked-test gates remain incomplete. See the
[measured engineering report](../research/general-engineering-report-44-47.md).
