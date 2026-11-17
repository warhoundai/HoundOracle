# Larger live Gemini comparison — provisional, incomplete protocol

Executed 2026-11-16 after explicit $8 authorization. Submitted inference implementation was frozen at the prepared campaign04 manifest; citation hardening added later was **not** silently inserted into the running job. Source checkpoints: `349dd63`, `4c88efb`, `fc1aa6b`; the submitted code SHA and cohort SHA are in `../campaign-04/next-cloud-preflight.json`.

## Design and execution

48 fresh source-disjoint FEVER components, 16 YES / 16 NO / 16 MAYBE, excluded from all previous training, validation, calibration, evaluation, invalidated traces and cloud cohorts. Labels came from original human annotations cross-checked against the gold-evidence mirror, never Gemini-generated labels. Three paired repetitions, seeded randomized arm order. Labels appear only in the controller's scoring records, not inference prompts. Model pretraining exposure remains unknown.

Systems: standalone Gemini with supplied evidence; the same Gemini behind FP32 MiniLM + logistic evidence evaluation, original calibrated .85 route, two inference threads. Evaluator caching disabled. Both arms have identical complete-corpus empty-evidence preflight and model output format. Exact model/version `gemini-3.8-flash`, LOW thinking, Google Gen AI SDK 2.34.0, Vertex global endpoint, temperature 0, seed 1907, 8192 input/4096 output caps. CPU job: us-central1, 2 CPU/2GiB, immutable pinned Python image, existing service account, no IAM grants, no grounding/GPU/public endpoint.

This is a supplied historical evidence workflow. It is not a Google Search grounded comparison, autonomous agent/MCP network workflow, current primary-record benchmark or S3 reuse experiment. Per-task timing includes Oracle evaluation where used, token-count request and model generation; startup, scheduling, installation, model download, recovery gap and orchestration are separate. The two worker segments used the same sequential inference policy and shared model settings; no successful query was rerun.

## Failures and completeness

The initial worker returned 196 task records, then a baseline Gemini generation for `fever:93342`, repetition 2, failed with HTTP 504. The runner's uncaught exception stopped execution. Its per-query timer and usage were lost; the failure was reconstructed from the observed traceback with unknown duration/cost and its full reservation retained.

A bounded continuation reused the same job and original embedded inference modules, skipped completed keys, and did **not** retry the failed query. It returned another 89 records before the remaining worker timeout. The last pair, `fever:77281`, includes one possible in-flight baseline request and one unexecuted Oracle-assisted task. Neither is dropped or counted as a correct MAYBE.

All 288 planned task slots are accounted for: **285 emitted results = 261 reported model responses + 24 local Oracle answers**, one observed provider failure, one uncertain in-flight request and one unexecuted task. There were 262 known generation attempts and at most 263 including the uncertain request, below 288. The planned three-repeat protocol is **incomplete**; reported comparisons are provisional and do not pass release gates.

The continuation timeout was 400 seconds after the first worker's conservative 792.145-second elapsed duration, giving a configured active envelope of 1192.145 seconds. Platform shutdown/control-plane delays add overhead: the sum of task-Started to execution-Completed timestamps is about 1206.39 seconds, and is not a compute invoice. The platform enforced the configured recovery timeout; no manual extension was made. Queue/provisioning and the diagnosis/recovery gap further extend batch wall time. Actual billed compute remains unverified.

## Results with failures included

Each arm has 144 planned task slots. Failed/unexecuted tasks count against workflow accuracy and definitive coverage; they are also reported separately from factual mistakes.

| Metric | Gemini | Oracle + Gemini |
|---|---:|---:|
| Workflow decision accuracy | 82.64% (119/144) | 85.42% (123/144) |
| Definitive coverage | 56.94% (82/144) | 58.33% (84/144) |
| Correct definitive / all | 53.47% (77/144) | 55.56% (80/144) |
| Accuracy among definitive answers | 93.90% | 95.24% |
| Failed/uncertain/unexecuted task slots | 2 | 1 |
| Reported model responses | 142 | 119 |
| Token list-price total, allowing unknown usage | $0.109581–$0.152589 | $0.08584875 |
| P50 empirical identification bounds | 2.255–2.317 s | 1.881–1.907 s |
| P95 empirical identification bounds | 10.515–17.733 s | 5.822–5.916 s |

Missing durations are left unknown. Quantile bounds allow them to be anywhere from zero to unbounded; these are **empirical identification bounds, not confidence intervals**. Missing usage ranges from zero to the full pre-call token reservation. No refund is assumed. P99 can have an unbounded upper limit and must not be represented as an exact favorable number.

Observed model-token cost reduction is **21.7–43.7%**, P50 reduction **15.5–18.8%**, and P95 reduction **43.7–67.2%**. All 261 response token totals reconcile with reported input + visible output + reported thoughts. Introductory prices were rechecked from official documentation: $0.75/$3.75 per million input/output tokens through 2026-12-31. No cache discounts are deducted. Known reported token total for both arms is $0.19542975, at most $0.23843775 using the two uncertain reservations. Compute, network, storage, logging, taxes and invoices are unmeasured; this is not an all-in cost result or a verified total bill.

## Statistical uncertainty and adoption decision

Question-cluster paired bootstrap accuracy difference: +2.78 percentage points, 95% interval [0.00, +6.25]. Coverage difference: +1.39 points, interval [−2.08, +4.86]. The predeclared 2-point coverage-noninferiority margin is not established by that interval. Zero in a bootstrap interval is not evidence of equality or a uniform correctness guarantee.

A second diagnostic propagates missing-value bounds through 5000 paired question-cluster bootstrap draws. Its outer 95% reduction intervals are:

- Model-token cost: **[6.6%, 63.5%]**.
- P50 latency: **[−2.1%, 31.8%]**.
- P95 latency: **[−14.6%, 89.0%]**.

The model-cost result supports a scoped benefit under the stated exchangeability/measurement assumptions. Latency intervals still admit a regression. They are diagnostics for a small residual historical cohort with an incomplete protocol, not formal guarantees under optional stopping or distribution shift. A statistically established 30% all-in cost reduction, 20% end-to-end latency reduction and all-category noninferiority have **not** been demonstrated.

Keep the scoped research runtime and evaluator-reuse options. Keep production's unsupported-population abstention and direct Gemini bypass. Do not promote the rejected heads, claim general superiority, or publish a production service. Future models must use a new locked cohort; these results cannot become an endlessly retuned test set.

## Category and citation findings

`label-strata.json` reports all truth-label strata. YES classification improved (87.5% → 93.75%); NO remained 72.9%; MAYBE improved provisionally (87.5% → 89.6%), with failures included. These are gold-label strata, not coverage of numerical, technical, adversarial or current-information task families. The MAYBE stratum avoided **no executed model calls** through local evaluation; its 47 versus 48 calls reflects the unexecuted slot. Its reported token costs are not consistently lower. Agreement across these three labels does not establish a benefit on every benchmark.

The 24 local answers cover only eight unique questions and all matched the human labels. That is useful routing behavior but insufficient to qualify all future local decisions. Existing model/correctness calibration remains explicitly population-scoped, not general production authorization. The complete Gemini hybrid has no separately qualified universal final-confidence calibration.

The separately replayed new exact-citation guard would block one baseline definitive response (`fever:196744`, repetition 2) and no routed model responses. That answer matched the gold label, but its quote did not exactly match the supplied snapshot. The served outputs and primary benchmark were not rewritten. This guard can trade coverage for verifiable citation integrity; it does not prove entailment. Independent human citation adjudication and source-independence checks remain outstanding.

## Required conclusions

1. **Cheaper?** Lower model-token cost on this scoped provisional workload: 21.7–43.7%; outer bootstrap interval [6.6%, 63.5%]. All-in cost advantage and the 30% target remain unproven. Token cost per correct definitive answer was $0.001073 for the hybrid versus $0.001423–$0.001982 for Gemini.
2. **Faster?** Observed P50 reduction 15.5–18.8%, P95 43.7–67.2%. Missing measurements are bounded, not imputed; statistical intervals still admit regression. Full agent and batch/startup economics remain unmeasured.
3. **Accuracy?** Higher observed workflow score, 85.4% versus 82.6%, not a general statistically established factual superiority claim. Definitive-answer accuracy was 95.2% versus 93.9% on the returned selections.
4. **Definitive and correct?** 55.6% of all planned hybrid task slots, versus 53.5% for Gemini. Local Oracle answered 24 slots, eight unique claims; broader local coverage remains limited.
5. **Central reuse?** No central/S3 economics measured. Local evaluator reuse improved CPU timings in campaign04; it was disabled here and cannot explain these live results.
6. **What helped?** The calibrated simple local route avoided 24 model requests, all locally correct on this sample. Exact evaluator reuse and two-thread runtime improved local CPU profiles. Richer context features and the quantized DeBERTa challenger failed their gates and were rejected. Citation/failure hardening improved verifiability/accounting without fabricating benchmark wins.
7. **NNUE/neural search?** No demonstrated benefit; neither was adopted or presented as tested superiority.
8. **Public release?** Production/service release gates remain closed: incomplete replication, uncertain latency and coverage noninferiority, missing all-in costs, incomplete final confidence qualification, and untested broad current/grounded/agent/citation/security workloads. No GitHub/OPS publication or existing-product changes.

## Next research priority

The largest remaining route limitation is that all insufficient-evidence cases still require Gemini. Investigate calibrated local abstention with a separately controlled false-abstention/coverage-loss bound, and improve semantic evidence understanding using appropriate independent supervision. Do not route by the benchmark's gold labels or simply lower confidence thresholds. Expand independently adjudicated current/technical/numerical/contradictory/adversarial data before adding NNUE or learned search. Provider deadline handling and durable per-query records are prerequisites for the next complete live run.

This campaign stopped at its authorized worker-time window and local experiment bound, not at a global optimum or universal success. Further paid execution requires a new authorized execution window; the existing approval was not a license for unlimited retries.

## Artifacts and cleanup

- `benchmarks/results/campaign-05/`: every planned slot, scalar summaries, uncertainty bounds, strata and separate citation-guard replay, without redistributed source text.
- `../campaign-04/`: frozen protocol, cohort/code hashes, authorization, resource limits, negative experiments and recovery plan.
- `initial-status.json`, `recovery-status.json`, `cleanup.json`: observed failures and task-owned job deletion; regional job list confirmed empty. Existing service account and APIs preserved; no public endpoint or new IAM grant.
- Raw source-text logs/payloads and full responses remain `/workspace/houndoracle-data/cloud-05-*` and `cloud-fresh-payload.json`, outside the Apache source distribution. Model weights stay outside the repository.
- Reproduce analysis: `scripts/finalize_cloud_fresh.py`, `scripts/analyze_cloud_comparison.py`, `scripts/export_cloud_results.py`, `scripts/compare_cloud_bounds.py`. Replay old inference only from its submitted manifest/commit; current source includes later citation/error handling and must not silently replace it.
