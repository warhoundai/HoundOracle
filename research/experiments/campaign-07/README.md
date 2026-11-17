# Campaign07: source context, controlled failures and benchmark repairs

Baseline `141a9c7`. This bounded local episode tests three modifications without changing the stable engine, spending new cloud money, publishing, or adopting C++. Architecture changes remain in the experimental namespace.

## Frozen experiments

1. Add supplied document titles before passages in the original encoder; fit the same logistic head on old development only.
2. Use the frozen campaign06 fine-tuned encoder with the same document context and head protocol.
3. Add four cheap title-overlap/completeness features to original encoder logits and fit a logistic head.

All use fixed C=1, seed1916, the existing dedicated calibration partitions and a fixed .85 operating threshold. Screen requires at least2 percentage points more correct definitive coverage, no accuracy loss and no selective-risk increase. Old validation is repeatedly used development evidence, **not fresh proof**. Distribution-shift challenges use explicitly retired campaign06 cases, never relabeled or represented as new heldout data.

## Repairs before promotion

Input auditing found12 blank spans in screening data. Initial title-prefix v1 could manufacture title-only premises from them. The input serializer now preserves blank spans, and a regression verifies this. Both title-prefix v1 results are withdrawn for promotion; corrected v2 recomputes affected raw predictions, reuses identical unaffected encoder logits, and refits with unchanged hyperparameters. Repaired-row IDs and source hashes are recorded. No threshold/label changes were made.

The first source challenge process was interrupted before any artifact because its candidate adapter was not wired to inference. That implementation error was corrected before interpreting results; only retired examples were involved.

A separate substantive benchmark fix stops reporting answer-label agreement as citation correctness. See [citation metric erratum](citation-metric-erratum.md). Frozen old reports are preserved but their proxy citation fractions are withdrawn. New citation metrics remain null until independent span review exists. Old cached timings are also suppressed in primary corrected screen summaries.

## Independent-data constraint

A further500 unused whole-page/subject/claim-disjoint examples were sealed excluding every previously observed component. They contain only9 definitive labels: risk3YES/2NO/295MAYBE; locked4YES/0NO/196MAYBE. These can assess a narrow abstention-shift problem, but cannot certify broad coverage or both verdict classes. Neither panel is to be consumed to manufacture a general accuracy claim. The manifest records immutable hashes and IDs; source text stays outside this Apache repository.

GCP inspection still returns reauthentication required. No new Gemini call, cost saving, all-in cost, or end-to-end latency advantage is claimed. Existing cloud spending authorization is closed. Model and dataset redistribution restrictions from campaign06 remain in effect.

## Outcomes after repair

| Candidate | Old validation correct definitive /480 | Retired balanced correct definitive /300 | Retired strict selected errors | Disposition |
|---|---:|---:|---:|---|
| Stable original .85 reference |72|46|2/12 (16.7%)|Preserve; not newly qualified for shift|
| Cheap title-feature head |90|63|4/15 (26.7%)|Reject|
| Original encoder + document context v2 |172|96|4/19 (21.1%)|Reject broad promotion|
| Frozen trained encoder + document context v2 |218|128|11/31 (35.5%)|Reject broad promotion|

Title context helps balanced examples but does not solve strict-page reliability. Small accepted samples give wide uncertainty; these diagnostics are not statistical proof of universal regression. The stricter point error gate is already missed. The corrected screen summary includes exploratory paired group-bootstrap quality intervals, clearly marked as reused validation evidence; retired challenge artifacts include one-sided exact error bounds. No new independent performance claim or production calibration follows.

Three controlled candidates failed a broadly safe improvement. Stop this bounded episode under the predeclared rule; do not consume the severely skewed new panels to disguise missing positive/refuting evidence. Fresh risk and locked files remain **unscored**. Next decisive research needs a more representative independently labeled cohort, source/entity-grounded semantic supervision, and a live paired Gemini comparison. The cloud connection still requires reauthentication; new spending is not inferred from a request to keep optimizing.

Retained implementation: the blank-span safeguard and the benchmark citation-metric correction, with132 passing tests on Python3.11 and3.12. All model variants remain isolated experiments; stable decisions are unchanged. This is not a perfect engine, a global optimum, or evidence of superiority across all benchmarks.
