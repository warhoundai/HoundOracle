# Actual Vertex comparison, 2026-11-15

Status: both comparisons completed; one intervening model-download failure retained; temporary Cloud Run Job deleted. Campaign stopped at 47 generation attempts.

The authenticated Google Cloud MCPP connection successfully executed the authorized bounded pilot in `project-f6e3c8ea-6b96-44a8-afa`. The exact model was **gemini-3.8-flash**, LOW thinking, global Vertex endpoint, official google-genai 2.34.0, temperature 0, seed 1729. No model substitution was needed. The container ran in us-central1 on one CPU/2GiB, using the existing narrowly scoped service account. No new IAM grants, GPU, grounding, or public endpoint.

This compares matched supplied evidence: standalone Gemini versus the experimental local evidence evaluator with Gemini fallback. It measures the combined local evaluation + countTokens + generation path. It does not measure live retrieval, an autonomous agent tool loop, network access to an externally deployed Oracle service, or startup-amortized all-in operating cost. Historical human FEVER labels are independent of Gemini; possible model pretraining exposure remains unknown.

## Initial results

Six development questions, two paired randomized repetitions. MLP routing at .80 matched standalone decision accuracy (83.3%) and coverage (50%), but linear-interpolated P95 was 4.204s versus 2.838s. A fixed existing logistic route at .85 was screened by development replay, then frozen before four fresh source-disjoint confirmation questions. Replay latency is not live performance. The runner's selection rule used a floor order statistic; final descriptive reports consistently use linear interpolation. Neither definition gives an initial tail-latency win.

| Fresh confirmation, 4 questions × 2 repeats | Gemini | Precision Oracle + Gemini |
|---|---:|---:|
| Decision accuracy | 100% | 100% |
| Definitive coverage / correct definitive fraction | 75% | 75% |
| Model calls | 8 | 4 |
| Token list-price cost, total | $0.004767 | $0.00278175 |
| P50 seconds | 1.571 | 0.706 |
| P95 seconds | 1.945 | 2.561 |

Observed token cost declined 41.6% and median latency declined 55.1%; P95 **increased 31.7%**. This is a tradeoff, not a general speed win. With four unique questions, the exact 95% interval for the probability of all repetitions being correct is [39.8%, 100%]. Repetitions do not create independent questions. Bootstrap question-cluster comparisons are supplied, with explicit small-sample caveats. No equivalence or superiority claim is warranted.

All 35 initial generation attempts (one smoke plus 34 benchmark calls) succeeded. Total calculated token cost was $0.020196. Every reported usage total reconciled with input + visible output + reported thoughts. Missing thoughts fields are recorded separately; they are not asserted to mean a measured zero. These prices are introductory list prices recorded in the config; discounts, invoice, compute, egress, logging and storage charges are unverified.

The one development error shared by both systems was an abstention on FEVER `4818`, labeled NO. Do not relabel it to inflate results. No wrong definitive answers occurred in this tiny pilot. Exact citation URL/quote matching checks integrity, not semantic sufficiency; independent human citation adjudication remains outstanding.

## Reproduction and artifacts

- `protocol.json`, `frozen-job.json`: first frozen cohort and code digest.
- `compact-protocol.md`, `frozen-job-supplemental.json`: predeclared bounded follow-up.
- `scripts/build_cloud_comparison.py` builds an ephemeral manifest without submitting anything. `--supplemental` requires the preserved first payload to exclude earlier sources.
- `scripts/analyze_cloud_comparison.py` validates labels, record uniqueness/count and usage accounting.
- `scripts/export_cloud_results.py` publishes source-text-free per-question records and question-cluster uncertainty.
- `benchmarks/results/campaign-03/initial/` contains measured summaries and all 40 task records.
- Full source-text/raw response logs and payloads remain outside the Apache distribution at `/workspace/houndoracle-data/cloud-03-*` and `cloud-comparison-payload.json`. See dataset licensing records. No credentials or model weights are embedded in release artifacts.

Commands for local analysis (no cloud calls):

```sh
uv run python scripts/analyze_cloud_comparison.py --logs /workspace/houndoracle-data/cloud-03-logs.json --payload /workspace/houndoracle-data/cloud-comparison-payload.json --output /workspace/houndoracle-data/cloud-03-analysis.json
uv run python scripts/export_cloud_results.py --analysis /workspace/houndoracle-data/cloud-03-analysis.json --output benchmarks/results/campaign-03/initial
```

Production confidence/provenance gates remain unchanged. Neither neural checkpoint is generally production-qualified. NNUE and learned search were not trained or adopted; no evidence yet justifies their complexity. Preserve the deterministic production baseline and latency-sensitive Gemini bypass. Public release gates have not passed.

## Required conclusions

1. **Cheaper?** The frozen precision route used 41.6% less model token list-price cost in fresh confirmation. That is $0.000348 versus $0.000596 per task, or $0.000464 versus $0.000795 per correct definitive answer, before all compute/network/storage costs. All-in savings are unknown; the 30% all-in target has not been established. Initial MLP development savings were only 0.65%.
2. **Faster?** Fresh-confirmation P50 improved 55.1%, P95 worsened 31.7%. No general speed claim, and the 20% end-to-end latency target has not been established. Provisioning, package installation and model-load startup are outside these per-task times and matter strongly for small ephemeral batches.
3. **More accurate?** Observed classification accuracy matched Gemini (83.3% development; 100% tiny fresh confirmation). This does not establish statistical equivalence or noninferiority. No definitive false answers were observed in this cloud sample; the larger prior local evaluation exposes errors and limited coverage.
4. **Definitive and correct?** The combined precision+Gemini workflow answered 75% of fresh confirmation tasks definitively and correctly. Oracle itself handled 50% locally; the rest fell back. The larger 600-question local research evaluation achieved only 22.8% correct definitive/all for FP32 logistic at its original .80 policy; do not generalize the tiny .85 router cohort to that population. Production's unsupported-population abstention policy is unchanged.
5. **Central knowledge economics?** Not measured. Both cloud arms received the identical frozen evidence; no S3 requests, synchronization, freshness checks, or live searches occurred. Model-call avoidance is local evaluation, not a central-cache hit. Earlier exact local reuse timings cannot establish central/S3 savings.
6. **What helped?** Development-only screening replaced MLP with an existing higher-precision logistic route, then demonstrated fewer Gemini calls and lower median/model-token cost on fresh questions. It did not solve tail latency. The compact-output experiment is separately reported; it applies the same format improvement to both arms. No post-confirmation tuning or retraining occurred.
7. **NNUE/neural-guided search?** Not demonstrated, trained or adopted. Existing evidence does not justify them. The local neural encoder plus simple supervised head is the tested experimental system; no learned search policy was used.
8. **Ready for public release?** No production/public-service release: grounded live retrieval, full agent workflow, all-in cost, larger independent current/technical data, citation adjudication, robustness and general calibration gates remain outstanding. The source implementation and transparent research artifacts are reviewable locally. OPS and public GitHub were not published or modified.

Use the precision router experimentally for the declared historical supplied-evidence population when model-call savings matter and fallback tail latency is tolerable. Use Gemini bypass when tail latency dominates or the calibration population is unsupported. Do not use these experiments to route arbitrary private/current questions without the production permission, provenance, freshness and confidence gates.


## Supplemental measured result and campaign closure

The first compact execution failed before any model call because the immutable Hugging Face model URL returned HTTP 429. A single bounded recovery respecting Retry-After succeeded. Both failure and final execution statuses are preserved. No provider rotation, new permissions, or increased generation cap was used.

Two new source-disjoint questions × three paired repeats; same frozen precision router and compact evidence-index format in both arms:

| Compact-format workload | Gemini | Precision Oracle + Gemini |
|---|---:|---:|
| Decision accuracy | 50% | 50% |
| Definitive coverage / correct definitive fraction | 50% | 50% |
| Model calls | 6 | 6 |
| Token list-price cost total | $0.00250275 | $0.00229650 |
| P50 seconds | 1.335 | 1.345 |
| P95 seconds | 2.652 | 1.782 |

All twelve outputs parsed and passed citation-index validation. Every definitive citation reconstructed an exact original source passage. Oracle abstained and called Gemini for **every** task, so the 8.2% token-cost difference and 32.8% P95 difference arise from model/service variability in these repetitions, not calls avoided or a demonstrated Oracle routing benefit. P50 was 0.75% worse. This is not a compact-vs-verbose causal comparison: the question cohorts differ, and the format was applied equally to both arms. Compact mode remains opt-in experimental; no production promotion or claim that it solved tail latency.

The shared error was abstention on FEVER `68459` (NO); the supplied passages did not persuade either verifier to refute the proposition. Profession labels can be nonexclusive in open-world factual verification. Preserve the independent benchmark label and count the error; do not force a NO or relabel after observing outcomes. FEVER label correctness and sufficiency of a particular supplied evidence subset are separate issues. This result also demonstrates why the earlier 100% four-question score cannot be generalized. The two-question exact interval is [1.3%, 98.7%] for all repetitions being correct. See `benchmarks/results/campaign-03/compact/` for every measurement and paired uncertainty.

Across the campaign: 52 task executions on 12 unique questions, 46 task generation calls + one smoke = **47 generation attempts**, all successful. Six tasks were answered locally. Total calculated model-token list-price cost was **$0.02499525**; compute/network/logging/storage/invoice remain unverified. All 47 response token totals reconcile. Three sequential executions reused one job definition: initial success, download failure before inference, bounded recovery success. Observed container-start to execution-completion durations were approximately 120 + 43 + 126 = 289 seconds, under the 600-second worker envelope (these timestamps are not a compute invoice). Provisioning/wall-clock delays are separate and are recorded in execution statuses.

The task-owned job was deleted and the us-central1 job list confirmed empty. Existing service account/IAM were preserved. Cloud Run API remains enabled; API enablement alone creates no always-on worker, and disabling a project-wide API could affect unrelated resources. No public endpoint, GPU, grounding, training cluster, GitHub publication or OPS change occurred.

Stop reason: completed the predeclared bounded follow-up, with only one generation attempt left and too few independent questions to justify another experiment. No statistical convergence/global optimum claim. Best demonstrated experimental route for the original supplied-evidence confirmation is FP32 MiniLM + logistic at .85 with Gemini fallback; preserve direct Gemini bypass for unsupported populations or tail-sensitive workloads. General production calibration remains unqualified. Next research priority is a larger independently adjudicated current/technical matched-evidence and grounded-agent comparison, including all-in costs, rather than more complex neural search.

Validation: 93 tests pass on Python 3.11 and 3.12; Ruff and sdist/wheel build pass. Source implementation/results checkpoint `f8b953f`, followed by the closure commit recording the completed supplemental measurement and cleanup.
