# Completed live matched-evidence Gemini comparison

Google Cloud access was restored and the user subsequently authorized a new campaign ceiling of$50GCP/$300AWS. Two separate$5GCP tranches were reserved. Earlier$3/$8 pilots remain closed. This experiment compares48 blind hash-selected, component-independent historical VitaminC queries ×3 repetitions: Gemini3.8Flash LOW, global Vertex endpoint, SDK2.34.0, temperature0, seed1930,8,192input/4,096output envelope. Same cohort was fixed before local outcomes. No question was dropped after seeing results.

## Observed results

| Metric | Gemini matched evidence | Calibrated local FP32 |
|---|---:|---:|
| All-decision accuracy |75.00%|66.67%|
| Definitive coverage |55.56%|37.50%|
| Correct definitive/all |42.36%|31.25%|
| Accepted error |23.75%|16.67%|
| Recorded P50 |2.070s|0.0504s|
| Recorded P95 |7.947s|0.0857s|

All144 repaired Gemini slots succeeded. Reported token list-price cost totals**$0.102543**; host/storage/transfer/credits/invoice adjustments remain unmeasured. The latency columns compare a local warm model to a remote API; they are not a full agent-workflow speedup. Three repetitions provide service variability, not144 independent label samples. Statistical uncertainty clusters all repeats by the48 source components.

The local calibrated accuracy difference is−8.33points, with descriptive paired cluster95% interval[−22.92,+6.94]points. Coverage difference−18.06points, interval[−33.33,−2.78]points. The raw local candidate classifier scored77.08%, but its28.13% definitive error cannot substitute for calibrated public decisions. No replacement-quality or universal superiority claim is justified. No threshold or policy was tuned on this cohort.

Raw-verdict diagnostics:90 paired slots both correct,15 both wrong,18 Gemini-only correct and21 Oracle-raw-only correct. Numeric-token claims favored Gemini on this small sample; the cohort contains no explicit lexical negation and does not qualify that category. Five questions changed Gemini decisions across repetitions despite fixed seed/temperature. Descriptive strata are overlapping regex tags, not adjudicated domain benchmarks.

## Failed first attempt and repair

The first isolated deployment omitted citations.py. MAYBE responses returned, but definitive responses encountered ModuleNotFoundError after model inference. The execution was cancelled rather than counted as factual failure evidence or silently retried. All planned first-attempt slots, known charges and conservative unknown-charge bounds are retained in attempt-1/. The corrected replication includes that dependency, an isolated definitive-answer check, a regression, and a halt on packaging errors. Questions, prompt, policy and settings stayed fixed. Repair costs belong to engineering/failed-experiment accounting and are not hidden from the campaign ledger.

Citation-export format was also corrected: public-format traces retain URL, quote hash and character count; original source passages stay in private logs. Exact citation integrity is not independently audited entailment. Decisions, usage, failures and timings are unchanged; export-sanitization.json records the format correction.

## Scope and next gate

These are conditional human labels on supplied historical revisions, not current-world truth, live retrieval, central knowledge economics or a whole autonomous-agent workflow. Neural weights and source passages remain private research artifacts, separate from Apache source licensing. Campaign18 prepares actual colocated FP32 verification plus Gemini fallback with the unchanged policy; it remains blocked by a later connector-session expiration. Spending authorization persists, so another budget request is unnecessary. Existing product infrastructure remains unchanged; public-release gates remain closed.

Analysis: scripts/report_public_gemini_pilot.py and scripts/analyze_public_gemini_errors.py. Outputs: benchmarks/results/campaign-14/. Private raw responses: /workspace/houndoracle-data/public14-repaired-logs.json. Do not rerun the paid comparison to recreate analysis.
