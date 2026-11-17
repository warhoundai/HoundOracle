# Campaign 06: local training, failed architectures and safe routing

Baseline: `3b1783a`; original implementation `9123fdb` remains preserved. No new paid cloud calls, infrastructure changes, public publication or production model replacement were performed in this campaign.

## Measured experiments

| Candidate | Evidence | Decision |
|---|---|---|
| Original-model calibrated insufficiency selector | No validation examples selected at .80/.90/.95/.975/.99 | Reject: no useful savings |
| Per-passage pooled logistic aggregator | Validation accuracy 52.29% vs 52.92%; selected error 11.40% vs 8.93% | Reject |
| One-epoch MiniLM fine-tuning, 2,000 old development examples | 433.58 CPU seconds; real weight update; ONNX max absolute parity error 2.75e-6 | Evaluate frozen policy independently |
| Fine-tuned verifier, .85 policy | Balanced heldout correct definitive answers 117/300 vs 46/300; strict-page heldout selected errors **13/32 (40.625%)** vs 2/12 | Reject general definitive-answer promotion |
| Original-NLI agreement plus conservative regime calibration | Retired balanced coverage 12.5%, strict coverage 1% | Reject before consuming successor heldout |
| Specialist only for insufficiency | Existing .85 definitive path unchanged; fine-tuned model may only produce MAYBE in an explicitly complete, proposition-validated evidence corpus | Experimental; see successor records when available |
| C++ rewrite | 97.79% of 60-query profile inside existing native ONNX `run`; all other work about .626 ms/query | No implementation: profile does not justify it |

The fine-tuned model improved old validation and balanced evidence understanding but failed source-disjoint reliability. This is an important negative result, not a production improvement. Insufficiency is not evidence of falsehood. No new trainable component can bypass privacy, spending or provenance rules.

## Data and methodology

Official FEVER human labels were crosschecked against a pinned public gold-evidence mirror. New training-partition research inputs were acquired outside the repository. Original source text, large weights and full evidence traces remain outside the Apache source distribution. FEVER/Wikipedia and mirror attribution/share-alike obligations require redistribution review; no AVeriTeC was used.

Two regimes are reported separately: strict whole-page/subject/claim components (strong exclusions but residual roughly 90% NEI), and balanced fact/span/subject/entity components (shared background pages possible). Balanced aggregate accuracy cannot establish performance in the strict regime. No datasets were relabeled after failures. All previously scored fresh risk/test cases were retired before successor work. Successor construction attempts failed on class availability before writing/scoring; size amendments are recorded in `protocol.json`.

The additional pinned mirror `test.jsonl` has SHA256 `3f4c9a9c6919f9a08af2ad2bcb9cc2be77a656169d93ff69a68fbd750c677b9e`, 6,626,459 bytes, 9,847 accepted unique cases crosschecked against official shared_task_dev, zero IDs overlapping the earlier mirror validation cohort. Components are recomputed across these inputs; zero ID overlap alone is insufficient leakage protection.

## Crash safety

A private SQLite task journal reserves maximum calls and token cost **before** SDK requests. Completed tasks are skipped, unknown interrupted requests retain reservations, configuration/provider/tenant bindings are immutable, and recovery never automatically repeats uncertain calls. A real subprocess abrupt-exit regression verifies this behavior. Optional short-lived S3 signed URLs support bounded private checkpoint snapshots with conditional writes, integrity validation and fail-closed conflicts. This backend is tested with controlled transports; it has not been deployed to AWS or independently exercised against live S3 in this campaign.

## Limits and next experiment

GCP and AWS inspection returned reauthentication errors. Earlier paid campaign authorization is closed; no new expenditure is inferred from the request for continuous work. Thus no new Gemini quality, all-in cost, grounding, central reuse economics or end-to-end latency result exists here. Prior campaign05 remains provisional and incomplete. Current evidence does not justify a broad public-performance release.

Next paid experiment should compare Gemini-only and the unchanged definitive verifier plus insufficiency specialist, using a fresh sealed workload and the durable journal/checkpoint runner. Source/entity grounding and negation handling are higher-priority quality investigations than a native-language rewrite. Training is reproducible via the frozen source, dataset hashes, seed/config, safe weights and export parity artifacts; weights remain local research artifacts.

## Successor one-shot results

No thresholds changed after scoring. All1,200 successor cases were evaluated once; ground truth was not passed to inference. The baseline .85 definitive path remained exactly preserved.

| Panel | Questions | Additional Gemini calls avoidable | False abstentions | Local P50 / P95 |
|---|---:|---:|---:|---:|
| Balanced risk |300|21 (7%)|1|.0878s / .1579s|
| Strict-page risk |400|75 (18.75%)|0|.0941s / .1996s|
| Balanced locked |300|24 (8%)|1|.0918s / .1890s|
| Strict-page locked |200|43 (21.5%)|0|.0912s / .1652s|

These are **additional** skips after the original definitive path; no new definitive answers are introduced. One supported balanced locked question was withheld. The strict residual has91.5% MAYBE, so its higher skip rate cannot be generalized. Per-question records and confidence intervals are in `benchmarks/results/campaign-06/safe-specialist-*`. Balanced aggregate intervals use separately stratified Bonferroni bounds, not an iid pooled-binomial fiction. Small class samples remain underpowered for some narrow coverage-loss guarantees. The specialist remains opt-in research, not a new production default.

### Answers to the product questions

1. **Cheaper than Gemini?** Not established all-in. Extra live calls appear avoidable on these frozen corpora; actual token savings require paired provider execution. Prior campaign05 token-only bounds remain separate provisional evidence.
2. **Faster?** No new end-to-end P50/P95 comparison. The table reports local pipeline time, not improvements over Gemini.
3. **More accurate?** Fine-tuned definitive verification helped balanced evidence but materially failed strict-page reliability. It was rejected for broad use.
4. **Correct definitive percentage?** Rejected fine model:39% balanced and9.5% strict heldout. Original model:15.33% and5% respectively on those earlier panels. The safe successor adds zero definitive answers and preserves the existing path. These are local research panels, not live-search product coverage.
5. **Central reuse economics?** Unmeasured; checkpoint persistence is not central knowledge reuse.
6. **What helped?** Fine-tuning improved balanced semantics; restricting it to insufficiency yields measured skipped-call opportunities. Durable reservations address repeat-call/campaign interruption failures. No new live Gemini economic benefit is yet proven.
7. **NNUE/neural search?** Not trained or adopted; no demonstrated advantage. Pooling and agreement guards were rejected. Native inference profiling does not support a C++ wrapper rewrite.
8. **Public release justified?** No performance claim or broad verified release passes the current gates. Preserve the baseline and reproduce the next paired experiment first.

### Why this bounded episode stopped

Multiple controlled candidates failed to provide a broadly safe Pareto improvement. The remaining promising specialist now needs independent **live workflow** evidence; both cloud access and fresh spending authorization are unavailable. Continuing to tune on scored heldout sets would invalidate them. All changes, including rejected models, are retained explicitly as research. No always-on background execution is claimed. Next useful work is source/entity-aware evidence compilation and a durably checkpointed paired provider comparison, not an unsupported universal performance claim.
