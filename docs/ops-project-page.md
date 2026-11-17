# HoundOracle — draft OPS project content

Publication status: draft. Do not replace the current OPS website or change any
released project listing. Add this as a dedicated HoundOracle page only after the
research-preview publication gate passes. Keep HoundHelios separate.

**Evidence-grounded verification infrastructure for AI agents.**

HoundOracle collects traceable evidence, checks supported factual relationships
and returns YES, NO, MAYBE or INVALID. It preserves the exact question's polarity
and abstains when a definitive answer is not justified.

- Run offline with local evidence and SQLite.
- Self-host with your own retrieval providers and credentials.
- Optionally synchronize explicitly approved public evidence snapshots.
- Integrate through Python, authenticated REST or local MCP.

Architecture: question compiler → local evidence memory → bounded retrieval →
provenance/dependency grouping → semantic compiler and symbolic checks → evidence
aggregator → calibrated decision gate.

Current scope is a development baseline for controlled release records and typed
quantity comparisons. Unrestricted natural-language factual verification, trained
neural evaluators and real-world qualification remain in development. No accuracy,
cost or latency advantage over conventional LLM research has been established.

Before publication, replace this paragraph with the verified repository URL,
versioned documentation, runnable integration examples, benchmark commands and
raw result links. Clearly label synthetic tests separately from independent
benchmarks. Use the namespace verified through OPS; do not assert unverified
legal ownership relationships between WarHound AI and Croxira.
