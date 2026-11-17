# Independent per-question evidence verification

## Why a separate protocol

The earlier VitaminC experiment asked whether every accepted answer in a2-4-query contrastive case was correct. HoundOracle's public decision answers one proposition. Those losses differ: a case-any-error probability is stricter than per-question error. Campaign08 failures are retained unchanged. This experiment does **not** claim to solve the all-pairs/case problem by changing its denominator.

Pinned human-labeled VitaminC real revisions, same rights/attribution as campaign08. Independent page/case/claim/span components are formed globally. Calibration uses1,000 unused queries for fitting and1,000 further independent queries for risk. Validation is207 reused development queries. The locked test is600 **new** component-uniform queries from unused official test components; one hash-selected query per component. None of these calibration/test components were gradient-trained. Exact hashes/IDs and scope are in `dataset-manifest.json`.

Maps are fitted on fit only, selected by validation Brier, then checked with one-sided exact binomial bounds on independent selected queries, Bonferroni alpha.05/5. Unlike the previous correlated case rows, each risk row is its own component here. Bounds apply only under this declared query-population/exchangeability assumption, not arbitrary web research or individual truth guarantees. Minimum confidence remains80%. The research pipeline now fails closed on invalid bounds, truthy nonboolean certificates and confidence caps below80%; legacy valid bootstrap certificates remain supported.

## Untouched600-query results

Selection was frozen as `continued-mixture` before any locked predictions. No post-test winner substitution.

| Frozen system | Raw accuracy | Definitive coverage | Correct definitive / all | Accepted errors |
|---|---:|---:|---:|---:|
| Original encoder/frozen head |62.33%|25.67%|22.17%|21/154|
| Prior FEVER-trained encoder |64.33%|18.33%|16.33%|12/110|
| First mixed-data encoder/frozen head |73.83%|33.67%|30.00%|22/202|
| First mixed encoder/replay head |74.17%|34.00%|30.67%|20/204|
| Continued encoder/replay head, preselected |**74.33%**|**37.00%**|**33.83%**|**19/222**|

The preselected model correctly answered203/222 definitive outputs (**91.44%**). Its family-adjusted one-sided heldout error upper bound is13.93%, below20%. It still gets19 accepted answers wrong; it is not perfect.

Against the original matched-context reference, raw accuracy improves12points, paired95% interval[8.0,15.83]points, Holm-adjusted exact McNemar p=2.24e-8. Versus the strongest prior FEVER encoder, the gain is10points,[6.33,13.5], adjusted p=3.26e-7. The second training stage's raw-accuracy difference versus the first mixed encoder is **not statistically established**: the new head control is74.17% versus74.33%, with a paired interval including regression. Continued training does improve observed correct-definitive coverage by3.17points over that control; its descriptive paired interval is[1.5,5.0]points. This interval is not a family-wise guarantee for all reported metrics.

Per-question outcomes, exact intervals, paired comparisons, calibration diagnostics and failures are in `benchmarks/results/campaign-09/`. Evidence text remains in the separately downloaded licensed dataset. Raw/refusal metrics are both reported; MAYBE responses are never excluded to inflate accuracy. All systems use the same256-token, one-thread context budget. These are semantic subsystem results, not a replay of the production512-token default engine.

## Practical interpretation

Two actual CPU training stages processed6,401 example exposures, incorporating3,201 unique real revision pairs plus1,600 historical FEVER examples and replay. First-stage training delivered most of the raw accuracy gain; additional training delivered smaller raw gains and better selective coverage. Training from scratch, NNUE, MCTS, graph evaluators and C++ wrappers were not needed for this gain.

The chosen model is usable through the existing experimental `ResearchPipeline` with its exact trusted manifest, head, runtime configuration and population-specific calibration artifact. It is an offline conditional-evidence verifier. It does not replace the default engine or qualify arbitrary live-retrieval confidence. Model weights remain local research artifacts; license compatibility review is needed before weight/data redistribution.

## Product questions

1. **Cheaper than Gemini?** Unmeasured.37% local definitive coverage is an opportunity to avoid fallback calls, not a measured all-in saving or proof of equal Gemini quality.
2. **Faster?** No new Gemini/agent latency comparison. Local timing is studied separately in quantization campaigns; contended score-collection times are withheld as comparative latency.
3. **More accurate?** Yes against the original/prior FEVER semantic references on this independent conditional query population; no Gemini superiority or general-world truth claim.
4. **Correct definitive percentage?**33.83% of all queries;37% answered definitively;91.44% precision among those answers.
5. **Central reuse economics?** Unmeasured. No private questions or public evidence contributions were uploaded.
6. **What helped?** Real contrastive supervision, replay, equally supervised controls and a risk controller with the right independently sampled unit. The original case-any-error failures remain real.
7. **NNUE/search gains?** Not demonstrated or adopted. Quantization targets the already-native runtime bottleneck separately.
8. **Public release?** No broad performance release: real retrieval, agent+Gemini economics, independent citation review and broader domains remain unqualified. GCP access was subsequently restored; campaign14 comparison is prepared pending fresh spending authorization. No new cloud charges were incurred during these local experiments.
