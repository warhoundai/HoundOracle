# Continued optimization, 2026-11-16

Baseline checkpoint: `09a940f`. Six predeclared local experiments executed without new cloud charges. The production verifier, existing products and OPS were preserved. Research options remain explicit; no universal benchmark-win claim.

## Actual results and decisions

| Experiment | Evidence | Outcome |
|---|---|---|
| E01 empty permitted corpus | 480 validation questions × 3 passes; replay of all 52 prior cloud tasks | No empty corpora in either workload; zero calls saved, zero verdict changes. Retain as a narrowly scoped abstention fast path; reject as a demonstrated optimization for these workloads. |
| E02 bounded exact evaluator reuse | 480 questions × 3 interleaved paired passes, 62 source components | Local P95 105.47 → 81.65 ms (22.6% observed reduction), P50 51.57 → 0.166 ms; all 1,440 candidate verdicts identical to baseline. Retain opt-in evaluator reuse for repeated frozen evidence. |
| E03 lexical/numeric-context logistic head | Train 2,000 development questions; separately calibrate and validate; no locked/fresh evaluation | Accuracy 52.9% → 51.7%, definitive coverage 23.3% → 21.5%. Selective error 8.93% → 8.74%. Reject: reduced coverage and classification quality despite slightly lower accepted error. Checkpoint preserved as rejected research. |
| E04 two inference threads | 120 balanced validation questions × 3 randomized paired repetitions | Encoder P50 48.92 → 30.88 ms; P95 100.64 → 75.53 ms (24.95% observed improvement). Zero logit/verdict drift. Retain scoped runtime option; no general production promotion. |

The evaluator-reuse workload includes the first pass that populates the cache. Its first-pass P95 was 102.17 ms; later-pass P95 was 0.235 ms. Warm hits were 960/1,440 by construction, not a predicted production hit rate. The paired source-cluster bootstrap interval for overall P95 reduction is [11.6%, 34.7%]. This supports a local improvement on this workload, not a statistically established minimum 20% gain across workloads. Default scope remains unchanged; larger independent evaluation is still required.

The threading bootstrap interval for P95 reduction is [1.1%, 39.9%]; its bootstrap mean is 19.5%, distinct from the observed 24.95% ratio. A follow-up identity check covered **all 1,680 existing validation/calibration rows** and found bitwise-identical candidate, logit, raw-score and truncation tuples. The runtime-port artifact retains the original correctness mapping and risk certificate, changes the explicit thread configuration, and remains `production_qualified=false`. No locked evaluation or fresh cloud cohort was opened for scoring. Two cores can increase compute cost and alter throughput; no all-in cost advantage is inferred.

The .85 precision route's validation definitive coverage was only 16.0%, with 6.49% selective error. These are local verifier measurements. The profiling fallback marker always abstains and never simulates Gemini quality, provider cost or provider latency. Do not present the resulting 47.5% local-plus-marker classification accuracy as the quality of the real Gemini hybrid. Fallbacks required stayed 1,209/1,440 for all local arms: evaluator reuse saves CPU, not generative calls.

## Safety and scope

- Exact evaluator cache keys include unchanged question text, complete evidence including revision/freshness metadata, explicit permission scope, full calibration artifact and model configuration. No approximate semantic answer reuse.
- A bounded LRU and TTL apply. Results are deep-copied to avoid mutation poisoning. Evidence, polarity, date text, permission, policy or model changes miss the cache. Instances are private to the caller; do not share a router across concurrent tenants. Gemini answers are not cached, and an earlier Oracle answer never becomes evidence.
- Empty-corpus abstention requires both an explicitly complete permitted corpus and an already validated proposition. It does not classify missing evidence as INVALID or NO, does not suppress live retrieval, and does not apply to bypass mode. Unvalidated input follows the existing path.
- Contextual lexical/number features are statistical inputs, not grounded numerical verification or proofs. The rejected head never replaced the simpler checkpoint.
- Citation correctness still requires independent entailment adjudication. Labels agreeing with verdicts are not a citation audit. Candidate head evaluation records unknown citation quality explicitly. Cached encoder timing in head-training diagnostics is not newly measured end-to-end latency.

## What remains unproven

| Benchmark/task family | Current evidence |
|---|---|
| Repeated historical supplied evidence, local CPU | Measured reuse and threading improvements |
| Cold Gemini with/without Oracle | Prior tiny pilot had a P95 regression; new candidate not yet compared live |
| Accuracy/coverage on fresh held-out questions | Fresh cohort frozen; not scored or used for tuning |
| Current documentation, releases, dates/quantities, multi-hop | No broad independent cloud win demonstrated |
| Ambiguity, adversarial content, provenance/privacy | Targeted regression tests; no comprehensive measured dominance |
| Live retrieval and grounded Gemini agent workflow | Not executed in this campaign |
| Central S3 reuse and total operating cost | Not measured; no savings claim |
| NNUE or learned search | Not justified by current evidence; not adopted |

A universal win across these families is not an acceptance result we can honestly assert. Router selection should eventually be driven by demonstrated category-specific quality, coverage, cost and latency. Statistical noninferiority and superiority need larger independent samples; passing tests alone cannot supply that evidence.

## Prepared next comparison and blockers

`next-cloud-preflight.json` freezes 48 fresh source-disjoint components, balanced 16 YES / 16 NO / 16 MAYBE. It excludes all prior train/validation/calibration/evaluation, invalidated scoring and cloud cohorts. The selection uses labels and length limits, never model predictions. Full source text remains outside the Apache source distribution; IDs, source hash and protocol are committed.

The prepared, locally tested manifest is `/workspace/houndoracle-data/cloud-run-fresh.json`. It compares the frozen FP32 logistic .85 route using the two-thread scoped runtime against Gemini 3.8 Flash LOW on identical evidence and output format, three randomized paired repetitions. No answer caching in either arm and evaluator caching disabled for this cold comparison. Identical empty-evidence preflight in both arms prevents a deliberately inefficient baseline. At most 288 model calls, $6.193152 worst-case model-token reservations, one temporary 2-CPU/2-GiB worker with a 1,200-second timeout, $0.50 compute allowance, no retries/grounding/GPU/public endpoint/new IAM grants. Proposed overall authorization: **$8 pay-as-you-go**, with credits/billing unverified. Startup/model downloads and cleanup are explicit; current prices/access must be rechecked after authentication.

The read-only project check returned `requires reauthentication`. The old $3 pilot is closed. The user explicitly authorized the prepared new $8 campaign on 2026-11-16. The connector subsequently reauthenticated and the frozen comparison was submitted as `houndoracle-fresh-20261116-lccf9`; no additional spending confirmation is needed within that scope. No new paid model calls or cloud resources were created. The next cohort is still historical FEVER, so even a win there would not establish all-category superiority.

## Reproduction

- `scripts/benchmark_router_optimizations.py`: paired local profiling and priming-inclusive summaries.
- `scripts/replay_empty_corpus.py`: clearly labeled counterfactual archived-token accounting, applied to both cloud arms; no fabricated live timings.
- `scripts/train_context_head.py`: single prespecified rejected head experiment, safe JSON checkpoint, separate calibration, no locked evaluation.
- `scripts/profile_threads.py`, `scripts/qualify_thread_runtime.py`: paired CPU performance and full existing-calibration/validation tuple identity checks.
- `scripts/prepare_next_comparison.py`: one-time fresh cohort freeze, refuses overwrite.
- `scripts/build_fresh_comparison.py`: prepare and compile an immutable-image job manifest without provisioning.
- `scripts/cloud_fresh_runner.py`: bounded fixed-policy runner, tested locally with provider stubs. Provider stubs are tests, never benchmark evidence.

Measured per-query results: `benchmarks/results/campaign-04/`. Full source text and encoder scoring/training traces stay outside the source distribution in `/workspace/houndoracle-data/`. This campaign is bounded by local time and missing cloud authorization/access, not statistical convergence or a global optimum.

Validation: 109 tests pass on Python 3.11 and 3.12; lint and sdist/wheel build pass. The larger runner's locally executed fixture tests verify paired record completeness, shared preflight and the 288-call hard limit. These tests are not provider benchmarks.

Official model pricing was rechecked on 2026-11-16: introductory $0.75/$3.75 per million input/output tokens through 2026-12-31. Retrieval hash is in the next-cloud preflight. Billing/credit availability remains unverified; the user authorized pay-as-you-go within $8.


## Additional isolated encoder screen

While the live comparison remained frozen, E05 screened pinned `cross-encoder/nli-deberta-v3-small` (`fa2804872c3b4bd748f38c0185cc85775361e735`), using its published AVX2 quantized ONNX file. Model card declares Apache-2.0 and SNLI/MultiNLI training; base/training-data rights remain separate. Downloaded 181.2 MB including tokenizer/config/card, verified the upstream LFS SHA256, no Torch/GPU/executable remote code, no weights in the Apache source distribution.

On the prespecified 120 balanced validation questions, raw accuracy was **50.8% versus MiniLM FP32's 60.0%**. The challenger failed the screen and was rejected before head training, calibration or fresh/locked evaluation. This does not isolate model architecture from quantization, and is not a general DeBERTa assessment. Model-card benchmark claims did not establish a benefit for Oracle's supplied historical evidence. No cloud policy changed. The model license/revision/file hashes and the negative screen are preserved.


## Citation-contract hardening (separate from frozen cloud execution)

E06 adds an exact source-snapshot citation guard to future matched-evidence Gemini adapter calls. YES/NO requires at least one citation; every cited URL and nonempty quote must occur in the supplied corpus. Unsupported/altered citations return MAYBE with `UNVERIFIABLE_CITATION`, retain paid usage, and are never treated as INVALID input or silently counted as a correct abstention. Grounded Google Search responses are outside this particular guard. Exact quote integrity still does not establish entailment or source independence.

Seven targeted tests cover omitted citations, invented URLs, fabricated quotes, empty spans, malformed citation objects, capitalization and the distinction from semantic support. All definitive Gemini citations in the archived campaign03 traces passed this contract, so the archived replay shows no accuracy/cost improvement; this is safety hardening. The submitted campaign05 remains exactly its original frozen implementation. Future source code uses the new guard by default, and any guard replay of the submitted outputs must be reported separately. No live result is retroactively rewritten.

## Provider failure and bounded continuation

The submitted live job emitted 196 successful task records, then Gemini returned 504 `DEADLINE_EXCEEDED` on baseline question `fever:93342`, repetition 2. The original runner's uncaught exception aborted the worker. That benchmark instrumentation flaw was repaired: future runners emit each per-query error, retain its reservation, abstain with an internal failure code, and continue within their existing cap. Regression tests cover this case.

A sequential continuation reuses the same temporary job, original embedded inference modules, original question/arm order and seed, same model, same calibration, and zero evaluator caching. It skips every completed query and **does not retry the failed call**. The first 178 reported model calls plus one uncertain failed generation consume 179 original reservations. At most 91 remaining task calls can run, so cumulative attempts cannot exceed 270 of the authorized 288. The first worker's conservative observed elapsed time was 792.145 seconds; the continuation has a 400-second timeout, limiting cumulative worker time to at most 1,192.145 seconds, below the authorized 1,200-second envelope. No new IAM grants, additional job definition, public service, GPU or grounding.

The failure's exact per-query duration and token usage were lost when the original runner crashed. They remain unknown, with the full $0.021504 token reservation retained; no cost refund or invented duration. Reports show conservative token-cost bounds and empirical latency-quantile identification bounds when observations are missing. A missing value suppresses unqualified point estimates/paired intervals. Any unexecuted tasks at the resource cutoff are reported explicitly rather than dropped. This continuation does not expand the user's $8 authorization or change the benchmark's factual decision policy.

## Final live campaign status

The frozen larger comparison stopped at the bounded recovery timeout with one pair incomplete. All 288 planned slots, errors and uncertainties are preserved. See [campaign05](../campaign-05/README.md) for actual results: a scoped model-token cost benefit, observed latency improvements with wide uncertainty, and release gates still closed. The temporary job is deleted.
