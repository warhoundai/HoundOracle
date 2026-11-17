# Local training, logical views and calibration follow-up

The99%target is not met. The best new development result was a small CE-only
attention adapter:179/207correct(86.47%), versus176/207(85.02%)for frozen29. It
corrected3cases without new errors on this panel, but failed the declared2-point
material-gain gate. It remains a private research checkpoint, not a production
replacement. The independently validated29configuration and default API remain
unchanged. These are reused-development classifier scores, not public API accuracy.

|Fixed development policy|Correct /207|Raw accuracy|
|---|---:|---:|
|Frozen29|176|85.02%|
|CE-only512-pair adapter|179|86.47%|
|Same adapter with pairwise ranking|177|85.51%|
|Original/negated-view averaging|173|83.57%|
|Three-class affine score adjustment|174|84.06%|
|CEcontinuation on1536additional pairs|174|84.06%|

The CEpilot's exploratory paired95%gain interval was0–3.38percentage points;
contrastive versus CEwas-2.42–0points. All intervals on this heavily reused panel
are exploratory, not independent qualification. More training regressed versus
the pilot; source labels were not changed or examples removed from evaluation.
No rejected policy received a fresh held-out score or public confidence setting.

## Actual training and artifacts

Two512-pair/1024-row arms trained100,611parameters each: rank8query/value factors in
the last4attention layers and the classifier. All base weights were frozen. Both
completed128updates, in207.81and199.35seconds. A separately declared CEcontinuation
completed384updates on3072rows from1536additional training source groups in566.21s.
The optimizer/RNG were reset explicitly; the continuation is not proof of a causal
benefit from data quantity alone. Some examples were seen by the original base.

All supervision came from previously available human real-revision training data.
Median pilot pair word overlap was92.86%; selected pairs had no identical encoder
inputs with conflicting labels. Training sources did not overlap validation or
calibration. Initial base-logit parity error1.67e-6; warm-start parity0. Checkpoint
hashes, version pins and every loss step are preserved. Safe adapter files are
about0.4MB each outside the source repository; the large base was not copied or
modified. Actual parameter changes were measured, not simulated training.

The negation experiment scored207new explicit negated views and used one fixed
mean; it did not tune a wrapper/weight sweep. Views are correlated, not independent
evidence. Three-class calibration used separate500-case class-map and500-case
correctness-map groups. It made no useful accepted-answer gain. A score adjustment
is not a factual-correctness guarantee.

## Fresh evaluation safeguards

Before38's result, prepared1000fresh risk and500fresh locked source/event components.
Exclusions connect pages, cases, exact claims and passages and remove all previously
used components. Selection did not use predictions or label filtering. These cohorts
remain unscored because38failed development. Historical citations now use explicit
revision IDs; upstream passage matching and independent span adjudication remain
unavailable. Data text stays private with its separate attribution/share-alike terms.

A new regression test caught the assessment's locked-read ordering before any fresh
scoring. The risk certificate is now checked before reading locked results. Changed
model rejection also passes. No previous held-out result was changed or invalidated.

## Practical interpretation

No new independent overall accuracy,99%precision, live-search reliability, central
reuse saving, NNUE benefit, whole-agent Gemini advantage or release gate was proved.
The pilot's raw correct-definitive fraction was129/207(62.32%); base127/207(61.35%).
These unqualified raw candidates do not bypass production provenance, calibration
or supported-claim requirements. The default general-claim API limitation remains.

All work ran locally under2CPU/8GiB with zero new AWS/GCPspending. Cloud connections
require reauthentication; existing300USD AWS/50USD GCPauthorization remains valid,
and reservations are unchanged. Local electricity/host cost is unmeasured. There
is no new cloud worker to clean up and no public publication. A separate bounded
local reasoning-first comparison is tracked in campaign40; its outcome is not
represented by this report. Stable rollback and all rejected artifacts remain.
