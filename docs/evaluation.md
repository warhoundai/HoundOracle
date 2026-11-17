# Evaluation protocol and component gates

Protocol version 1, recorded before external experiments. Local synthetic smoke
results are engineering checks, not independent accuracy evidence. No architecture
superiority claim is currently permitted.

## Splits and comparisons

Keep training, development, final-policy calibration and locked test partitions
separate. Group by original claim, paraphrase family, originating source and time.
Fit logistic/NLI models on training only; choose model/settings on development;
calibrate the final stopped policy, including retrieval and abstention, on
calibration only. Evaluate once on locked test, with model/corpus/provider revisions
and exclusions recorded. A changed policy requires recalibration.

Compare A deterministic, B FIRE-style, C two evaluators, D incremental aggregator,
and E graph/policy research under matched corpora, retrieval budgets, supported
claim types and confidence requirements. Separate paper-style B (including
zero-search and forced verdicts) from evidence-required B. Separate live/frozen
retrieval, cold/warm cache, and optional Gemini fallback. No Gemini integration is
qualified yet. Do not convert FEVER NEI or HoVer NOT-SUPPORTED into refutation.

Primary measurements: selective error and coverage, evidence sufficiency and exact
citation validity, retrieval evidence recall, extraction accuracy, complete per-query
reserved/actual costs, p50/p95/p99 wall latency, peak memory and CPU. Include
conflicts, copied reports, unfamiliar claims, expired sources and negative controls.
Use Brier score, reliability plots and ECE with bin counts and alternative binning;
ECE alone can hide subgroup or class-specific miscalibration.

Report paired cluster bootstrap intervals (2,000 resamples by claim/source family)
for cost and latency at matched risk and coverage. For error estimates, report
counts and appropriate binomial/cluster intervals, explaining assumptions. A
confidence threshold is not a statistical guarantee. Calibration bins must be fixed
before calibration outcomes are inspected. Sequential per-step calibration does
not automatically survive optional stopping; evaluate the entire stopped policy.
Conformal/risk-control claims require their own assumptions and implementation.

## Measurable component gates

| Component | Gate |
|---|---|
| Question/compiler/cache | All supported semantic regression tests pass; zero false cache equivalences in the adversarial test set; unsupported semantics abstain |
| Retrieval/security | No private-network access in SSRF/redirect tests; request/cost caps hold under concurrent reservations; failures never yield NO |
| Evidence store | All correction, expiry, ancestor revocation and new-contradiction tests invalidate affected verdicts; exact spans round-trip |
| Semantic evaluator | Report span/entity/date/unit precision and recall on independently annotated eligible data; production requires >=99% grounded extraction precision with lower 95% bound >=95% |
| Aggregator | Compare trained logistic versus heuristics at fixed retrieval; report dependency/conflict subgroup errors and calibration before promotion |
| Risk controller | Minimum estimated correctness .80 plus evidence gates; locked-test upper 95% selective-risk bound <=.20 and coverage >=.50 on the explicitly declared eligible population |
| API/MCP/jobs | Auth, isolation, cancellation, cumulative budgets, failure/restart and protocol tests pass; no secret leakage |
| S3/sync | Opt-in, rights filtering, signature/hash tampering, expiry, private dependency and revocation tests pass; staging restore drill before deployment |
| Advanced evaluator/policy | Target improvement >=10% in cost or p95 latency, or >=5 percentage points in coverage at fixed risk; paired 95% interval excludes zero improvement; risk worsens by <=1 percentage point; all gates retained |

Advanced targets are engineering acceptance decisions, not measured performance.
No parameter or architecture is promoted based only on synthetic results. For NNUE,
require incremental/full recomputation agreement within 1e-6 float tolerance and
<=1 percentage point decision disagreement after quantization, independently of
its speed gate. Reject the experiment if CPU encoder/network cost dominates and
end-to-end gains cannot meet the threshold.

## What ran locally

`uv run python benchmarks/run.py --output benchmarks/local-result.json` generates
400 calibration examples and 160 disjoint-entity test examples from controlled
release-record templates. The default uncalibrated engine has zero definitive
coverage. A synthetic-only calibration artifact exercises YES/NO, abstention and
cache paths. Template sharing makes ordinary binomial intervals optimistic; the
JSON flags this explicitly. These are smoke results, not proof of real-world
factuality, production calibration or savings against an LLM.

`benchmarks/fire_style.py` independently implements the broad iterative control
loop with injected model/search functions and reservation accounting. It replaces
SBERT similarity with exact query repetition and does not reproduce FIRE prompts,
retry behavior or datasets. Its mocked tests are not a FIRE benchmark. A live
reproduction remains blocked on licensed evaluation material and a metered provider
configuration; do not spend cloud funds to conceal these missing prerequisites.
