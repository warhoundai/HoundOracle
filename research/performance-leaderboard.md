# Performance leaderboard

Historical FEVER supplied-evidence task, final 600 questions. No Gemini results.

| Configuration | Coverage | Correct definitive / all | Accepted accuracy | Decision |
|---|---:|---:|---:|---|
| Preserved production core | 0% | 0% | undefined | Keep stable; unsupported population |
| Raw NLI + calibration | 0% | 0% | undefined | Reject as useful calibrated verifier |
| FP32 NLI + logistic | 26.5% | 22.8% | 86.2% | Retain research checkpoint |
| FP32 NLI + MLP | 30.8% | 26.3% | 85.4% | Retain research checkpoint |
| Int8 NLI + logistic | 34.5% | 28.8% | 83.6% | Heldout risk interval exceeds 20%; no promotion |
| Int8 NLI + MLP | 29.8% | 25.5% | 85.5% | Faster candidate; repeated quality/coverage gate failed |

No production replacement, no all-in cost ranking, and no general superiority claim.
See engineering-report.md for intervals, latency definitions and missing comparisons.

## Actual matched-evidence Gemini pilot (campaign 03)

Fresh confirmation only: 4 questions × 2 repeats, gemini-3.8-flash LOW.

| Workflow | Accuracy | Definitive coverage | Token cost total | P50 | P95 |
|---|---:|---:|---:|---:|---:|
| Gemini with supplied evidence | 100% | 75% | $0.004767 | 1.571s | 1.945s |
| Precision logistic Oracle + Gemini | 100% | 75% | $0.00278175 | 0.706s | 2.561s |

41.6% token-cost reduction and 55.1% median reduction; 31.7% P95 regression.
Tiny sample; no statistical equivalence/general superiority, live retrieval or all-in cost claim.
See [actual execution report](experiments/campaign-03/README.md).

Separate compact-format supplemental cohort: 2 fresh questions × 3 repeats.

| Workflow | Accuracy | Coverage | Token cost total | P50 | P95 | Model calls |
|---|---:|---:|---:|---:|---:|---:|
| Gemini compact | 50% | 50% | $0.00250275 | 1.335s | 2.652s | 6 |
| Precision Oracle + Gemini compact | 50% | 50% | $0.00229650 | 1.345s | 1.782s | 6 |

Oracle forwarded every supplemental case. Differences in model cost/P95 are not
call-avoidance gains. Both abstained on one NO-labeled question. No production
promotion or causal compact-vs-verbose claim. Total campaign 47 generations,
$0.02499525 estimated model-token list price; all-in cost unknown. Temporary job deleted.

## Campaign 04 — local improvements, separate from live Gemini results

| Experiment/workload | Baseline P95 | Candidate P95 | Observed reduction | Verdict changes |
|---|---:|---:|---:|---:|
| Exact evaluator reuse, 480 questions ×3 including first pass | 105.47 ms | 81.65 ms | 22.6% | 0/1,440 |
| Two CPU threads, 120 questions ×3, encoder only | 100.64 ms | 75.53 ms | 24.95% | 0/360 |

Local CPU timing only. No measured new Gemini end-to-end or monetary saving.
Warm hit rate is constructed. Context-feature head lost accuracy/coverage and was
rejected; empty-corpus optimization hit no actual cases. Full results, uncertainty
and the unfunded next comparison are in [campaign 04](experiments/campaign-04/README.md).

## Campaign 05 — larger live comparison, incomplete planned replication

48 fresh source-disjoint questions, 144 planned slots per arm. One provider 504,
one uncertain request and one unexecuted task retained; bounded continuation
reused the same policy. This is **provisional**, not a release-gate pass.

| Metric | Gemini | Oracle + Gemini |
|---|---:|---:|
| Workflow accuracy including failures | 82.64% | 85.42% |
| Definitive coverage | 56.94% | 58.33% |
| Correct definitive / all | 53.47% | 55.56% |
| Token cost bounds | $0.109581–$0.152589 | $0.08584875 |
| P50 empirical bounds | 2.255–2.317s | 1.881–1.907s |
| P95 empirical bounds | 10.515–17.733s | 5.822–5.916s |

Observed token reduction 21.7–43.7%; paired outer bootstrap diagnostic interval
6.6–63.5%. Observed latency improves, but its intervals admit regression.
All-in costs, universal confidence, broad task-family gains and the 30% cost /
20% end-to-end latency targets remain unestablished. Coverage noninferiority is
not established at the predeclared 2-point margin. See
[campaign05 report](experiments/campaign-05/README.md) for failures, bounds,
strata, exact interpretation and cleanup.

## Campaign 06 — local research only

The trained MiniLM model improved balanced correct-definitive coverage (117/300 vs46/300) but failed strict-page selected risk (13/32 errors); rejected as a general verifier. Passage pooling and an NLI-agreement guard were also rejected. No new Gemini/all-in performance measurement; the stable definitive route remains unchanged. A MAYBE-only specialist is under separate successor evaluation. Profile evidence does not justify a C++ rewrite. [Experiment and limitations](experiments/campaign-06/README.md).

## Citation metric erratum — campaign07

Legacy local NLI/head `citation_supported_definitive_fraction` values derived from FEVER verdict-label agreement are withdrawn as citation-quality measurements. Accuracy/coverage results remain unchanged. New metrics require an explicitly recorded independent span review; otherwise citation quality is unmeasured. See [erratum](experiments/campaign-07/citation-metric-erratum.md).

## Campaign07 — source-context challengers rejected

Correct definitive counts on reused validation improved from72/480 to172/480 (title-context original) and218/480 (title-context trained). Retired strict-page selected error remained21.1% and35.5%; the cheap grounding head reached26.7%. No production promotion or new independent/Gemini performance claim. Input and citation-reporting bugs were repaired;132 tests pass on both supported Python versions. [Detailed outcomes](experiments/campaign-07/README.md).

## Campaign09 — independently evaluated contrastive training

On600 untouched, component-uniform historical claim-evidence queries, preselected continued MiniLM achieved74.33% raw accuracy versus62.33% for the original encoder/frozen head (paired95% gain interval8.0–15.83points). Definitive coverage37%; correct definitive/all33.83%; accepted precision91.44% (19/222 accepted errors). Family-adjusted one-sided error upper13.93%. This qualifies the declared conditional research population, not arbitrary live web research. [Full report](experiments/campaign-09/README.md).

## Campaigns10–13 — preserve FP32

Unsigned, per-channel signed and mixed-precision quantization reduced measured local latency, but failed fidelity/accuracy/coverage gates. Transformer FP32 fusion preserved logits but failed the speed gate. All rejected; no quantization locked test was scored. Strongest validated local research checkpoint remains campaign09 FP32. No NNUE or custom C++ performance advantage demonstrated.

## Campaign14 — comparison prepared, not measured remotely

Restored GCP access; prepared48 fresh questions ×3 repeats with SDK/version pins and hard application reservation limits. New$5 spending authorization pending. Local cohort results and all planned remote slots are preserved; no new Gemini or all-in benefit claimed. [Status and limitations](experiments/campaign-14/README.md).

## Campaign14 — completed live Gemini comparison

144/144 repaired Gemini3.8Flash LOW requests succeeded across48 independent historical evidence queries ×3. Gemini all-decision accuracy75.0%, coverage55.6%; calibrated local FP3266.7%,37.5%. Accepted error23.75% versus16.67%. Raw local candidate77.08% is not a calibrated accepted-answer result. Reported Gemini token cost$0.102543; all-in and agent savings remain unknown. Failed initial packaging attempt is retained separately. [Report](experiments/campaign-14/README.md).

## Campaigns15–17 — three cheap challengers rejected

Nonlinear MLP heads, a fixed checkpoint midpoint and verdict-conditional calibration all failed declared development gates. No locked data consumed and no production replacement. Preserve the campaign09 FP32 research checkpoint; no further tuning on this reused validation episode.

## Campaigns18–19 — active authorized follow-up

Actual colocated verifier+Gemini fallback is prepared; GCPsession expired before deployment. One bounded AWS CPU training run uses6,401 new human real revision pairs plus replay (9,601 total); initial checkpoint and fresh independent risk/test cohorts sealed before training. No success claim until qualification. Campaign reservations are tracked separately from unknown invoice charges.

## Campaigns19–20 — larger training and identity correction rejected

One real9,601-example CPU training epoch completed in924.334seconds. Raw reused-development accuracy76.81% versus77.78%; provisional correct-definitive fraction32.37% versus37.68%. Failed gate, no fresh risk/test consumed. Motivated fixed identity-head ablations also failed. Stable checkpoint unchanged; worker and its temporary IAM/profile/group cleaned up. [Complete engineering/product answers](engineering-report-20261116.md).

## Campaigns27–31 — component screens, not product qualification

Dual-encoder stacking and a larger generic DeBERTa model failed reused-development
gates. A separate supervised DeBERTa adaptation is recorded in campaign29.
A retrieval-context experiment recovered raw accuracy65.22→77.78% by using one
passage instead of concatenating three, with model/packing P50/P95 reduced from
144.95/253.85ms to63.42/144.46ms. Raw definitive coverage fell75.36→67.63%.
This is a scoped research improvement on a constructed historical corpus, not a
Gemini, open-web, all-in-cost or99% result. The public API abstained on every general
claim in this diagnostic; broader compiler/qualification work remains necessary.
The stable research checkpoint and production confidence policy are unchanged.

## Campaign29 — independent risk, fixed policy

A real6400-example DeBERTa adaptation passed development and the untouched1000
source/event risk cohort: raw82.9%; fixed80% policy78.0% all-decision accuracy,
51.4% definitive coverage,49.1% correct definitive/all,491/514correct accepted.
Conditional one-sided correctness lower93.36% at alpha.025. Strict99 accepts zero
and fails. The sealed600reference comparison follows this gate. No new Gemini,
all-in economics, live-search or production qualification is implied.

## Campaign29 — sealed600 confirms research improvement

Fixed80% policy: adapted DeBERTa76.50% overall verdict accuracy versus stable60.17%;
53.67% coverage versus36.33%;298versus200correct definitive answers. Paired95%
accuracy-gain interval12.50–20.33points; coverage-gain interval13.33–21.50points.
Accepted precision92.55% (24errors/322accepted). Raw accuracy84.0%. Retain as the
strongest independently tested research quality/useful-answer configuration, while
preserving1921rollback and default API behavior. The99% goal still fails; public
compiler support, independent citation audit and live workflow economics remain
unqualified. This cohort is now consumed and must not become a tuning set.

## Campaigns32–34 — generators rejected; live Gemini uncertainty retained

On48balanced reused development cases, adapted DeBERTa35/48correct; local4B32/48;
local30B-A3B30/48. Both open-weight generative replacements rejected. On the same
24-case subset, NLIraw19/24; GeminiLOW18/24,HIGH20/24. HIGHgain interval-8.33to25points,
P50/P95request6.978/30.795s versus2.186/8.291sLOW. No independent superiority or99claim.
All48Gemini and96local observations preserved, temporary compute/IAM/groups cleaned.
Strongest independently qualified research configuration remains29; no default API
change or public release. See engineering-report-campaign-32-34.md.

## Campaigns35–39 — small development signal, no qualified replacement

Frozen29raw176/207(85.02%); CEadapter179/207(86.47%); contrastive177/207; negationmean173/207;
class-score adjustment174/207; CEcontinuation174/207. All failed declared development
promotion gates. Real128/128/384training updates and safe small checkpoints retained;
base29unchanged. Fresh1000risk+500locked sealed and unscored. No new independent,
whole-agent, live-search or99result. See engineering-report-campaign-35-39.md.


Campaign40: local R1 direct4/12, reasoning5/12, existingNLI8/12; failed preregistered gate. Campaign41: post-hoc deterministic relation replay6/12 vs5/12; correct definitive5 unchanged, no independent qualification. See [report](engineering-report-campaign-40-41.md). No new cloud spend; no production promotion.


Campaign42 CLOSED: restored GCP, user pilot target95% overall. Gemini3.8Flash HIGH paired24 reused development questions,48attempts all reconciled. Baseline17/24(70.83%), audit18/24(75%); historical localNLI19/24(79.17%). Baseline2provider504 failures retained as errors; audit0failures. Correct definitive11/24both, coverage62.5%/58.33%; failed improvement gate,95%notmet, no promotion. CandidateP50/P95 7.311/31.962s vsbaseline5.224/53.690s, broad uncertainty. Known model token subtotal$0.23873625 excluding2unknownusage failures; complete/all-in costsunknown. Jobdeletedand0remainingjobsverified; prior14jobsalso deleted. GCPreservations26/unreserved24, AWS72/unreserved228; invoicesunknown. Fresh39untouched. See [report](experiments/campaign-42/report.md).


Campaign43: new deterministic primary-record tool matches112/112mechanically derived release-index references,96correctdefinitive/112,16MAYBE. Scoped feature/retrieval pilot only, not comparable with general NLI/world truth accuracy or a95%generalqualification. ColdSDKP50/P95 95.07/143.37ms,warm13.46/21.10ms; no modelcalls/newcloudspend, hostcostunknown. Source-date semantics conflicts retained; no baselineweights/globalpolicychanged. See [report](experiments/campaign-43/report.md).


LATEST CLOSED STATE — campaigns 44–47: opt-in general-proposition verification,
local semantic evaluator, bounded full-source retrieval, dependency grouping and
policy-bound research calibration implemented. Campaign46 AWS adapter rejected:
389/500 versus baseline394/500. Worker terminated; own IAM/profile/SG deleted.
No active worker from this campaign. Campaign47 raw validation391/500; research
selection226/244 correct at48.8%coverage. Accepted API remains unqualified, zero
definitive coverage. Latest controllerv4 intentionally invalidates archivedv3
research calibration. New source-separated candidates500dev/500val/1000cal/
1000locked plus7816training reserved; independent audits and end-to-end world
labels pending. Locked qualification tests UNSCORED.95% target NOT MET.
AWS84reserved/216unreserved; GCP26/24, actual invoices unknown. No public release.
See [engineering report](general-engineering-report-44-47.md). Supersedes ACTIVE46.


Campaign48: fixed freshness-after-limit starvation and upstream freshness in FTS.
Controlled synthetic retrieval40cases:10/40before,40/40after; NOT factual accuracy.
Medianquery0.105ms→0.238ms. Policy changed; previous qualification invalidated.
Public project page/Pages workflow prepared. GitHub CreateRepository denied;
Hostinger requires reauthentication. Nothing published.95% remains unmet.
See experiments/campaign-48/report.md and ../docs/publication.md.
