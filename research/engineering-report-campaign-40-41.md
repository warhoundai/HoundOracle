# Local optimization campaigns 40–41

The 99% target remains unmet. These are small reused development comparisons of
historical supplied evidence, not independent live-search or public API results.
No new AWS or GCP expenditure was incurred; local host and electricity costs are
unmeasured. Both cloud connections require reauthentication. No public release.

## Campaign 40: local reasoning model, rejected

Pinned DeepSeek-R1-Distill-Qwen-7B Q4_K_M ran locally on two CPU cores with
llama-cpp-python 0.3.19 compiled with GCC. Twelve balanced development questions
were selected without filtering on model outcomes. Direct JSON scored 4/12;
reasoning followed by JSON scored 5/12; the existing NLI model scored 8/12.
The preregistered improvement requirement was two additional correct answers:
the candidate failed. Paired accuracy difference interval was −25 to +41.67
percentage points; this tiny sample cannot establish a general model ranking.

Direct P50/P95 latency was 15.175/18.881 seconds; reasoning-first was
121.358/153.952 seconds. These are this local evidence workflow's measured
latencies, not complete AI-agent timings or a Gemini comparison. All 24 dataset
slots completed without truncation. Two earlier compatibility attempts decoded
zero completions and are preserved separately, not counted as accuracy tests.
Actual formatter integration tests fixed unsupported callback forwarding and
preserved native stopping rules. The owner wrapper removed its 4.68 GB RAM model
checkpoint on completion; no worker remains.

## Campaign 41: deterministic relation preservation, investigate only

Error analysis found that a generative formatter could turn an explicit
“Missing” relation into NO. The experimental deterministic decoder maps explicit
supported/refuted/insufficient conclusions to YES/NO/MAYBE, checks exact echoed
propositions, and rejects ambiguous or conflicting labels. It does not infer
falsehood from lack of support. Production confidence remains unqualified.

This post-hoc replay scored 6/12 versus 5/12, with definitive coverage falling
from 100% to 83.33%. Correct definitive answers remained 5/12. Eight drafts were
normalizable; two verdicts changed. This is a development bug fix, not independent
accuracy evidence. No new model calls were made. Avoided formatter time is only
a counterfactual opportunity; candidate cold latency and savings are unmeasured.
Binding a single permitted passage is not an entailment audit.

## Retained configuration and next gate

Campaign 29 remains the strongest independently tested research configuration.
The fresh source-disjoint campaign-39 risk and locked cohorts remain unscored;
failed training candidates cannot open them. None of these experiments qualifies
99% overall or selective correctness, whole-agent Gemini superiority, central
reuse economics, NNUE benefits, or a public release. The next useful work is an
independent evidence/label audit and a fresh, bounded comparison after cloud
access is restored, rather than additional tuning on these development cases.

Protocols, configurations, failures and per-case metadata live in the campaign
folders. Private evidence text and reasoning drafts are excluded from git.

## Verification

Python 3.11 and 3.12: 223 passed, one optional Torch module skipped in each core
environment. The four low-rank tests pass separately in the Torch environment
with PYTHONPATH=src (an initial invocation without that path could not import the
editable project and was corrected). Ruff, staged whitespace checks and isolated
wheel build pass. A pre-existing Starlette deprecation warning remains. Base29
safetensors SHA256 remains
c520871a91b6497b628d60aad945b54c34819d8b53c5c3b0bf814fab7d3f3e70.
No stable model weights, public engine configuration, OPS site or released product
was replaced. These tests establish software regressions, not factual accuracy.
