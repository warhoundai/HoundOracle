# Run, benchmark, reject: local generators and live Gemini

The99%target remains unmet. This continuation executed96local open-weight model
predictions and48live Gemini requests, retained every outcome and rejected both
local architecture replacements. No validated accuracy improvement emerged from
these candidates. The strongest independently evaluated research configuration
remains campaign29; its raw84%and selected76.5%overall accuracy on its sealed600
cohort must not be confused with public API or arbitrary live-search accuracy.

## Same-evidence development measurements

|48balanced reused development cases|Correct|Raw accuracy|
|---|---:|---:|
|Adapted DeBERTa29|35/48|72.92%|
|Local Qwen3-4B-Instruct-2507 Q4_K_M|32/48|66.67%|
|Local Qwen3-30B-A3B-Instruct-2507 Q4_K_M|30/48|62.50%|

|24-case paired subset|Correct|Raw accuracy|P50/P95request seconds|Token list-price USD|
|---|---:|---:|---:|---:|
|Adapted DeBERTa29|19/24|79.17%|Not paired|0generative tokens; compute unmeasured|
|Gemini3.8Flash LOW|18/24|75.00%|2.186/8.291|.0267975|
|Gemini3.8Flash HIGH|20/24|83.33%|6.978/30.795|.11631|

The LOW/HIGHpaired95%bootstrap accuracy-gain interval is-8.33to25.0percentage
points. Two extra correct cases are not evidence of a reliable/general advantage.
One paired repetition per case is not a repeated latency qualification. All48
Gemini slots completed without failure, all reservations reconcile, exact model/
SDK/settings were pinned and labels/local predictions were not sent to Gemini.
Model token list price totals.1431075USD; host/storage/invoice costs remain unknown.
These are controlled supplied-evidence development comparisons, not whole-agent
live-search or independently selected final-test results.

## Implemented changes and actual fixes

- Added an explicitly unqualified local generative evidence compiler using an
  in-process open-weight model. Empty evidence returns MAYBE without invoking it;
  malformed responses and uncited definitive answers are failures. Original
  proposition/polarity is preserved as data. Schema validity does not prove truth.
- Used the C++llama.cpp CPU backend through its MIT Python bindings. Corrected a
  missing pkg-config build prerequisite after preserving the initial failed run.
  A C++backend and larger weights did not deliver an accuracy advantage.
- Added fixed-stage local and Gemini assessment scripts that reject incomplete,
  duplicate or unpaired observations. Per-question metadata and intervals are
  versioned; no failed/deferred query is silently dropped.
- Traced all48sampled claim/passage/label triples to the original human dataset;
  every transformation matched. Prepared blinded review for all48, not just model
  errors, with exact revision references and original-record hashes. No labels
  changed, rows removed, audited verdicts invented or source text redistributed.
- Added strict immutable Wikipedia revision references. A current page URL cannot
  establish what a historical revision said. A revision reference itself does not
  prove passage integrity, factual truth or semantic entailment.

Some disagreements involve ambiguous qualitative claims, missing entity-alias
context, or potentially questionable annotations. Original-source review confirmed
the labels were inherited, not that they are necessarily wrong. Models agreeing
against a label is not independent adjudication. The blinded audit remains pending.
Historical snapshot entailment and current-world truth require different targets:
web search may establish a fact that the supplied passage cannot establish.
Therefore live retrieval cannot honestly be called a historical NEI-label fix.
There are also genuine model errors; annotation questions do not explain them away.

## Required product answers

1. **Cheaper than Gemini?** No new all-in advantage established. Reported token
   charges are measured separately from unknown infrastructure/invoice costs.
2. **Faster?** No whole-agent P50/P95improvement demonstrated. HIGHreasoning was
   slower in the paired diagnostic. Local4B/30B inference P50/P95were2.189/2.747and
   4.285/5.723seconds on separate CPU deployments; do not treat those as causal gains.
3. **More accurate?** Both local generative candidates were worse than adapted NLI
   on the same48development queries. Gemini HIGHwas best on the24-case subset,
   with wide uncertainty and no independent-general superiority claim.
4. **Definitively correct?** On48, NLI26/48,4B24/48,30B23/48. On24, Gemini LOW11/24,
   HIGH14/24. These raw candidates are unqualified public decisions. The previously
   independent29research policy retains298/600correct definitive answers.
5. **Central reuse economics?** Not measured in these supplied-evidence experiments.
6. **Which changes helped?** No new validated semantic-accuracy improvement. Runtime
   repair enabled valid measurement; stricter output/provenance/audit tooling improved
   experiment integrity. Neither local generator replaced the stable configuration.
7. **NNUE/neural search/C++?** No new NNUE/policy advantage. Native C++inference was
   actually tested, but did not solve semantic correctness. No bespoke rewrite adopted.
8. **Public release?** Broad reliability gates still fail. General-claim public
   compilation, citation adjudication, current/live retrieval, calibrated coverage,
   factual accuracy and workflow economics remain unqualified. No publication occurred.

## Controls and retained work

Three temporary EC2workers were used: one installation failure, its repaired4B
worker, and one30Bworker. All terminated; both role/profile/policy/security-group
sets were removed; attached volumes were zero. The sole temporary GCPjob completed,
logs were exported and the job deleted. No active paid compute remains from this
continuation.203tests pass on Python3.11/3.12, lint and wheel build pass.

Authorized ceilings remain300USD AWS/50USD GCP. Total reservations72/22USD;
unreserved228/28USD. These are not actual costs. Billing API is disabled; no API
enabling, IAM changes on GCP, GPU or public endpoint was needed. AWSroles were
least-privilege temporary access to each experiment's private prefixes only.

This bounded candidate screen is complete without a passing replacement. The next
credible intervention is grounded evidence interpretation with independently
adjudicated truth/sufficiency and temporal context, followed by a fresh held-out
full-pipeline test. Reusing the consumed independent test or repeatedly scaling
these failed configurations cannot support a99%claim. No global optimum or
universal accuracy ceiling has been established. The99%goal has not been achieved.

See campaign32/33/34protocols, screens, metadata-only results under
`benchmarks/results/`, blinded audit preparation and docs/local-generative-experiments.md.
