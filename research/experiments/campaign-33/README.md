# Live Gemini thinking-level comparison

Restored Google Cloud access with explicit project/quota flags. Executed48real
Gemini3.8Flash requests:24balanced reused development questions, one LOW and one HIGH
call each, same supplied evidence/prompt/model/SDK/settings, randomized order.
All48planned slots completed, no failures or retries; reservation and response
records reconcile. The SDK is2.34.0, global endpoint, temperature0, seed1970,
max4096output/8192inputtokens and evidence-index JSON output. Gold labels and local
model predictions were not sent to Gemini. No Google Search grounding was used.

| Metric | LOW | HIGH |
|---|---:|---:|
| Correct verdicts | 18/24 | 20/24 |
| Definitive coverage | 58.33% | 70.83% |
| Correct definitive / all | 45.83% | 58.33% |
| Accepted error | 21.43% | 17.65% |
| P50/P95request seconds | 2.186/8.291 | 6.978/30.795 |
| Reported token list-price USD | .0267975 | .11631 |

Adapted NLI raw candidates scored19/24(79.17%) on these same questions. These raw
candidates are not calibrated accepted answers. Higher Gemini thinking achieved
two additional correct answers, but the paired95%bootstrap gain interval spans
-8.33to25.0percentage points. This small reused-development episode does not
establish significant improvement,99%accuracy, general Oracle/Gemini superiority
or all-in economics. Timings include the SDK's input count and inference requests,
not the full agent/search workflow. One paired repetition per question is not a
multi-run latency qualification. Citation indices were checked; semantic citation
support has not received independent adjudication.

The sole temporary Cloud Run execution succeeded in6m26.47s; logs were exported
and the task-owned job deleted. Billing API is disabled; no API enabling, IAM grant,
public endpoint or GPU was used. Credits and invoices remain unknown.4USDcampaign
reservation is not actual spending; model token list price totals.1431075USD.
Full configuration, exact per-question results and uncertainty are in screen.json
and benchmarks/results/campaign-33/metadata.jsonl. Source passages stay private.
