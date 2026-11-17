# Optimization history

- Baseline verified: `9123fdb`, 70 original tests pass; no repository or architecture restart.
- Preregistered bounded campaign; preserved original Git checkpoint.
- Added pre-abstention diagnostics, complete trace fields, unknown-cost handling and official Vertex SDK adapter with durable reservations.
- Dataset variant 1 invalidated: repeated IDs, class skew; raw failures and manifest preserved.
- Dataset variant 2: deduplicated, original human labels cross-checked, page/subject connected splits, independent untouched evaluation sealed.
- E01 pretrained fp32 NLI: validation accepted-error about 32%; confidence adjustment alone failed. Keep negative result.
- E02 correctness calibration: temperature/logistic/isotonic compared; raw encoder still fails risk gate.
- E04 exploratory supervised heads: logistic/MLP improve validation classification; scoped calibration enables selective research decisions. Safe JSON checkpoints retained.
- E03 int8: local latency reduction measured; quality/coverage regressions and uncertainty prevent production promotion.
- Final freeze and one-shot 600-question test: all candidate configurations reported; no tuning afterward.
- Exact local reuse measured on fixed repeated validation workload; no S3 economics inferred.
- Vertex pilot explicitly authorized ($3); submissions in two regions rejected for CPU quota before execution. No generation/worker usage occurred.
- Stopped for missing critical Gemini execution evidence and completed bounded local campaign. No public release, no global-optimum/convergence claim.
- Campaign 03: connector reauthenticated; actual Gemini 3.8 Flash LOW comparison executed on a temporary CPU Cloud Run Job. Initial MLP route lost P95; development-only replay selected existing precision logistic .85 before fresh confirmation. Four-question confirmation matched observed quality/coverage and cut token cost/P50, but P95 regressed. Full negative tail result retained.
- Predeclared supplemental compact evidence-index responses for both arms, two additional source-disjoint questions × three repeats. Same frozen router; remaining generation allowance only. No confidence/quality claims expanded from this pilot.
- Compact experiment: first execution failed on model-host HTTP 429 before any model call; failure preserved. One Retry-After-respecting recovery succeeded. Both systems 50% decision accuracy / 50% coverage on two fresh questions, all Oracle cases fell back. Different token usage/tail latency reflects model/service variability; no routing benefit demonstrated. Keep compact mode experimental, no promotion.
- Campaign 03 closed at 47 generation attempts and $0.02499525 estimated model-token list price; all-in charges unknown. Deleted temporary Cloud Run Job and confirmed regional job list empty. No public release. Preserved precision research checkpoint, production abstention policy and direct Gemini bypass. Stop at predeclared campaign boundary; larger independent evaluation remains necessary.
- Campaign 04 (2026-11-16): four controlled local experiments from `09a940f`. Empty-corpus shortcut saved zero calls on actual workloads; preserve negative finding. Scoped bounded exact evaluator reuse reduced priming-inclusive local P95 22.6% on 480 questions ×3, no verdict drift; no Gemini cost/latency claim.
- A single context-feature logistic head trained on 2,000 examples reduced accuracy/coverage and was rejected. No locked/fresh evaluation tuning or production replacement.
- Two inference threads reduced observed encoder P95 24.95% on 120 questions ×3; full 1,680-row validation/calibration tuple identity check passed. Retain scoped runtime port, production qualification remains false; compute-cost/throughput tradeoff unmeasured.
- Frozen 48 fresh source-disjoint questions and prepared/tested a larger matched-evidence Vertex runner. New campaign requires $8 authorization and connector reauthentication; read-only access currently fails. No new cloud spending or resources. Continue only after these blockers resolve; all-category superiority is unproven.
- User authorized the new $8 campaign on 2026-11-16. Permission persists; do not ask again for this prepared scope. Read-only access still requires connector reauthentication; no new cloud resources or calls executed yet.
- E05 isolated pinned DeBERTa-small quantized ONNX screen: 50.8% raw validation accuracy versus MiniLM FP32 60.0% on the same 120 questions. Reject before further training/fresh evaluation; preserves the download/license/hash record and avoids assuming a larger model is better. Running paid policy unchanged.
- E06 citation contract: matched-evidence definitive Gemini responses must carry exact verifiable URL/quote spans; otherwise fail closed with UNVERIFIABLE_CITATION and retain usage costs. Seven targeted regressions pass. Archived campaign03 has zero violations, so no historical performance gain claimed. Current live manifest unchanged; any new-guard replay stays separate.
- Campaign05 executed the authorized $8 / 48-question frozen comparison. Initial run emitted 196 records then provider 504; bounded sequential continuation emitted 89 more before its 400-second worker cutoff. Failed call not retried; one uncertain final request and one unexecuted slot retained. All 288 planned slots accounted for, planned three-repeat protocol incomplete. No extra resource window used.
- Larger provisional result: workflow accuracy 85.42% vs Gemini 82.64%, coverage 58.33% vs 56.94%, token-cost reduction bounds 21.7–43.7%. Outer paired bootstrap token interval 6.6–63.5%; latency intervals still admit regression and coverage noninferiority not established. No all-in/general superiority claim. Exactly 261 response usage totals reconcile; two uncertain attempts retain full reservations.
- Closed the execution window and deleted the task-owned job; regional job list empty. New broad/current/grounded/agent/citation/calibration work remains before release. Best retained research options are the simple scoped route, bounded evaluator reuse and two-thread CPU runtime; rejected candidates stay isolated. No global-optimum/all-benchmark success claim.

## Campaign 06 — real fine-tuning, rejected general promotion, safe specialist

Ran a controlled per-passage pooling experiment, real CPU MiniLM fine-tuning and export parity, fixed-threshold validation, independent risk/heldout comparisons, and a semantic agreement guard. Retained all failures. Fine-tuning substantially improved balanced examples but failed strict-page reliability (13/32 definitive errors), so it did not replace the stable verifier. A separate specialist is restricted to MAYBE only. Added durable task/provider reservations and conditional private checkpointing with abrupt-process-exit and conflict regressions. C++ was not justified by a profile with97.79% of local runtime already in native ONNX inference. No new paid calls or public release; live connections require reauthentication. See [campaign06](experiments/campaign-06/README.md).

## Campaign07 — three source-context candidates and benchmark integrity repair

At baseline141a9c7, tested supplied-title context with the original/frozen-trained encoders and four cheap title metadata features. Reused validation screens improved, but retired strict-page accepted-answer errors were4/19,11/31,4/15 respectively; none replaced the stable verifier. Found and repaired blank-span title-only premise construction before promotion; withdrew v1 results, corrected12 affected predictions and refit identical heads. Found legacy citation metrics conflating label agreement with independent citation review; separated fields and require recorded independent review before aggregating citation quality. No labels changed.132 tests pass on Python3.11/3.12. Sealed500 further source-disjoint cases but left them unscored:491 are NEI, so no broad coverage/accuracy certification is possible. Stop after three failed controlled candidates; live GCP reauthentication remains unavailable. [Report](experiments/campaign-07/README.md).
- Campaign09:600 untouched independent query components, frozen preselection,74.33% raw accuracy and37% definitive coverage; family-adjusted selective-risk upper13.93%. Contrastive training helped versus original/prior FEVER models; second-epoch raw improvement over first mixed-data control not statistically established. Full per-query records preserved.
- Campaign11: per-channel signed quantization reached about66% local speed improvement but failed quality/fidelity gates. Rejected, locked test untouched.
- Campaign12: final bounded mixed-precision masks (last1/2/3 blocks FP32) all failed98% label-agreement gate. End quantization episode without lowering gates; preserve FP32.
- Campaign13: native transformer fusion retained numerical parity but failed15% material speed threshold; rejected. No custom C++ rewrite justified by current evidence.
- Campaign14: restored read-only GCP access, verified existing minimal inference role and official model/pricing documentation. Prepared new48-component ×3 matched-evidence comparison with SDK2.34.0 and a disabled20-minute job. Fresh up-to$5 authorization pending; no paid execution. Improved failure/missing-latency metrics and tested complete log reconciliation. Central reuse and all-in economics remain unknown.
- Final harness hardening: failed MAYBE is incorrect regardless of gold label; unknown latency stays unknown; complete-slot Cloud Logging reconciliation and clustered repeated-query comparisons added. Future quantization noninferiority uses conservative exact marginal discordance bounds, retaining uncertainty even with zero observed discordances. No previously rejected model/test outcome changed.
- 2026-11-16: user authorized300USD AWS/50USD GCP. Recorded separate reservations before actions; no assumption that credits guarantee free execution.
- Campaign14 initial deployment omitted a dynamic citation dependency. Cancelled, retained failures/unknown charges, fixed bundle and isolated definitive-branch regression, then allocated a separate repair tranche. Repaired run completed144/144 requests. Gemini75% all-decision accuracy/55.56% coverage versus calibrated local66.67%/37.5%; replacement quality not established. Source text removed from public-format citation exports with an audited metadata-only correction.
- Campaign15 MLP16/32:78.26/77.29% raw development versus77.78% logistic; material-gain gate failed. Campaign16 equal-weight checkpoint average:77.29%, rejected. Campaign17 verdict-conditional map lost correct-definitive coverage, rejected. Stop this three-failure local episode; retain stable checkpoint.
- Campaign18 prepared actual hosted local encoder plus optional Vertex fallback, with paired randomized order and complete traces.8USD GCP/2USD AWStransfer reserved. GCPconnector expired before deployment; requested reconnection, not new spending permission.
- Campaign19: inspected AWS identity, available CPU/GPU quotas, no running/pending us-east-1 instances, Canonical image owner and0.357USD/hour c7i.2xlarge on-demand price. Created isolated encrypted expiring research bucket, no-ingress group and prefix-limited temporary worker role. Reserved10USD; launched one IMDSv2/encrypted-EBS/shutdown-terminate instance,45minute watchdog. One epoch on9,601 examples, new risk/test cohorts withheld. Preserve current best pending measured gates.
- Campaign19 completed actual9,601-example training/1,201steps in924.334seconds. Safe weights/export parity verified, worker auto-terminated and own IAM/profile/group deleted. Equally supervised head raw development76.81% versus77.78%; provisional useful-answer fraction32.37% versus37.68%. Reject before fresh risk/test; no stable replacement.
- Campaign20 used the observed encoder/head CE gap to motivate a fixed identity-head ablation on both encoders. No material raw/useful-answer gains; reject both without consuming fresh risk/test. Document full negative results rather than rescuing19 with a post-test selection.
- Static cloud-bundle import checking now catches deferred citation dependencies before paid deployment.154test-suite target after added guard, builds checked; metadata-only exports retain all outcomes without distributing licensed source quotes. GCPactual hybrid remains prepared/authentication-blocked; spending authorization persists. No active paid compute remains, private bucket expires retained artifacts after2days.


## Web23: local execution with internet discovery

Implemented self-hosted SearXNG and bounded source fetching after clarification. Local integration and security regressions pass; real EC2 upstream smoke returned Bing results but Google denied/DDG CAPTCHA. Retain as retrieval infrastructure, not a demonstrated accuracy improvement. Both temporary workers terminated and SG deleted. See research/experiments/web-23/README.md.


## Tools24: reuse public WarHound tools

Added HTTP adapters and challenge-aware retrieval. Validated original request contracts; restricted incompatible Scraper multi-page jobs. Retain as infrastructure/security improvement; no claimed accuracy gain. Google successful retrieval and automatic CAPTCHA solver remain unresolved. See research/experiments/tools-24/README.md.


## Captcha25: local text recognizers

Evaluated Tesseract and ddddocr on40 synthetic development cases. Retained alphabet case projection improvement for explicitly single-case fields (ddddocr15→25 correct), selected Tesseract default26correct and lower cold latency. No independent real-world or Google acceptance claim. No factual evaluator promotion. See research/experiments/captcha-25.


## Recaptcha26: Google v2 local solver integration, not qualified

Added a separately AGPL-licensed local Node/ONNX helper; fixed image-fetch guard and test harness compatibility, patched14high audit entries to0known vulnerabilities. Five bounded EC2 runs retained; valid image challenge attempts did not complete within120seconds. Keep experimental and stop this campaign, no reliable Google or factual-accuracy advantage claimed. All workers/temporary SG cleaned. See research/experiments/recaptcha-26.

## Campaigns27–31: supervised semantics and evidence packing

- Complementary encoder stacking failed the development useful-answer/coverage gate; rejected before fresh risk or locked evaluation.
- Excluded an ANLI-trained candidate after identifying upstream noncommercial terms. A separate Apache model without that training provenance reached only63.29% raw development accuracy versus77.78%; rejected. Its bounded EC2 worker and temporary group were cleaned up.
- Campaign29 runs one fixed6400-example supervised adaptation epoch on a separate temporary CPU worker; fresh risk/test inputs are excluded. Status and cleanup are recorded in its active-worker manifest.
- Corpus diagnostic: gold evidence ranked first for97.2% of supported/refuted development claims. Concatenating top3 passages reduced raw accuracy to65.22%. Selecting top1 recovered77.78% and reduced model/packing P50/P95 to63.42/144.46ms, but reduced raw definitive coverage. Retained only as an explicitly scoped, unqualified research adapter; no generic API promotion.
- Added a joint held-out99% goal that requires both overall accuracy and accepted precision bounds, a coverage floor and independently reviewed citations. Abstaining on hard valid questions cannot manufacture a passing score. No actual system has passed it.
- Campaign29 completed800updates/6400examples in1092.16seconds. Adapted DeBERTa reached85.02% raw reused-development accuracy versus77.78%; provisional80 correct definitive count103versus78. Passed development only;99threshold accepts zero. Retained and hash-verified the safe checkpoint locally. Worker terminated, own role/profile/policy/security group deleted and zero attached volumes confirmed. Independent fixed-policy risk evaluation opened only after freezing follow-up protocol; no production promotion.
- Campaign29 independent1000 risk:491/514correct accepted,51.4%coverage, one-sided correctness lower93.36% at alpha.025; passed fixed80 conditional research gate. Sealed600then scored once:76.5%overall verdict accuracy versus60.17%stable, coverage53.67%versus36.33%, correct definitive298versus200. Paired95%accuracy gain12.5–20.33points. Retain new research quality checkpoint, preserve stable rollback/public API. Strict99 still fails with zero accepted at99threshold; no citation/general-web/Gemini/all-in qualification. Do not tune against these now-consumed cohorts. GCPread-only project inspection still requires reauthentication; requested reconnection while completing local work.

## Campaigns32–34: local open-weight adjudication and real Gemini

- Restored GCPaccess using explicitproject/quota flags. Ran24paired development questions at LOW/HIGH,48requests, no retries/failures; all planned slots and reservations reconciled. LOW18/24,HIGH20/24; paired95%gain interval-8.33to25points. HIGHwas slower and costlier. Token list price.1431075USDtotal; all-inunknown. Logs exported and own Cloud Runjob deleted.
- Tested ApacheQwen3-4B-Instruct2507Q4_K_M through MITllama-cpp-python0.3.16CPU(C++backend). Initialworker missingpkg-configbuild dependency failed before inference; one bounded repair completed.32/48correct versus adaptedNLI35/48; reject.
- Controlled larger-capacity follow-up: Qwen3-30B-A3B-Instruct2507Q4_K_M, same48questions/prompt/schema/evidence,30/48correct; reject. PeakRSS31,136,648KiB. No fresh risk or locked data opened for either failed architecture.
- Both32workers and34worker terminated; ownrole/profile/policy/SG sets deleted and0attachedvolumes. Source archives private/expiring, all96local+48Gemini observations preserved as metadata. AWSreservations72/unreserved228;GCP22/unreserved28, actualinvoicesunknown.
- Prepared blinded upstream review of all48cases, not only errors; transformations allmatch originalhumanlabels. No labels changed or rows removed. Added immutable historical revision reference guards. Citation/annotation adjudication remains pending. Stable29and default API unchanged;99target unmet.

## Campaigns35–39: real local adapters and protected fresh evaluation

- Two128step LoRAtraining arms on512human revision pairs1024rows: CE179/207(86.47%)vsbase176/207(85.02%); contrastive177/207(85.51%). CEcorrected3without newerrors on reused development, but neither met frozen2pointgate. No publicpromotion.
- Negation-score symmetry173/207and full3class score adjustment174/207failed. Correlated views are not independent evidence; raw softmax scores are not truth probabilities.
- Continued CEpilot on1536additional training groups3072rows,384updates566.21seconds;174/207, rejected. Baseweightsunchanged; adapters~.4MBeachretained withsafehashes. No newcloudspend.
- Sealed1000fresh risk+500freshlocked beforecontinuationresult; none scored after failed38gate. Newguard regression caughtlockedread-before-riskcheck in the unusedassessment harness; corrected andtested beforefreshscoring. No labelschanged/evaluationrowsdropped.
- Cloudsessions expired; asyncreconnectionrequested, currentpaidbudgetauthorizationpersists. Separate localreasoning-methodexperiment40started, outcome pending. No global99claim.


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
