# Research dossier — initial method review

This is an initial source/method review, not a completed systematic review or
reproduction. Original full papers were retrieved and relevant methods, results,
ablations and limitations inspected; no research results below are Oracle results.
`source-manifest.json` records URLs and retrieved-content hashes. Downloaded papers
and research implementation files are outside the distributable repository.

## FIRE — highest priority

[Xie et al., Findings of NAACL 2025](https://aclanthology.org/2025.findings-naacl.158/),
§§3–6 and Appendices A–C; [implementation](https://github.com/mbzuai-nlp/fire),
`eval/fire/verify_atomic_claim.py`, `config.py`, `query_serper.py`, requirements.

The unified generative step chooses a binary verdict or one search query using
accumulated snippets **and parametric model knowledge**. Initially evidence is
empty; a zero-search verdict is allowed. Confidence is expressed through the
prompted choice, not a separately calibrated numerical correctness estimate.
Retrieved snippet strings are appended to context. The final stage forces a binary
judgment when search stops. Oracle cannot use this stopping rule for evidence-backed
YES/NO without a separately labeled changed protocol.

Published repetition control uses MiniLM sentence embeddings, cosine threshold .9,
with window two selected in the development experiment (Table 4); a diversity
prompt did not improve performance. The public code config is five steps, ten
retries, three search results, tolerance two. Its `<= max_retries` loop permits up
to eleven attempts at a step, and its search wrapper has another retry loop.
The snippet-comparison helper forces tensors onto CUDA despite an earlier CPU
fallback, so direct CPU reproduction needs a documented patch. Search calls lack
explicit timeouts. No code is copied because the repository has no declared license.

The paper's description mentions SerpAPI, while Table 7 and the inspected code use
Serper. Preserve this distinction in reproduction records. Table 1: development
Factcheck-Bench 631 claims; evaluation FacTool-QA 233, FELM-WK 184, BingCheck 142,
total 559. Supported and partially supported become True; refuted becomes False;
not-supported is excluded. BingCheck samples 100 of 3,581 True against 42 False.
FELM-WK retains unsplit claims. These transformations change the task and cannot be
silently adopted for Oracle's four public decisions.

Table 7 GPT-4o-mini costs: FIRE LLM $0.14/search $0.20; FacTool $1.49/$3.67;
FactCheckGPT $1.28/search not comparable; SAFE $0.43/$2.93. The headline 7.6x is
approximately `mean(1.49,1.28,.43)/.14`; 16.5x is `mean(3.67,2.93)/.20`, excluding
FactCheckGPT's separate scraping from the search ratio. These are aggregate costs
for those 559 transformed claims under the paper's providers/models/prices, not
per-query guarantees, universal architecture speedups or Oracle measurements.
Per-class precision/recall/F1 are reported; Oracle must additionally measure
citation sufficiency, abstention, correctness intervals and matched coverage.

Ablations include no-search, no-reason, at-least-one/two searches, inclusive prompt
and repetition windows. Extra searches did not automatically improve verification.
Error analysis examines 135 failed claims: dataset issues 44, knowledge issues 50,
reasoning issues 26, debatable opinions 15 (Table 8). This warns against treating
labels as infallible or model confidence as evidence sufficiency.

FIRE already supplies adaptive retrieval/verification and repetition stopping.
Oracle hypotheses are persistent evidence, source-dependency handling, deterministic
checks, local evaluation and abstention. They may reduce coverage or add overhead;
none is an established improvement. The independent callback-based control-loop
approximation in `benchmarks/fire_style.py` is tested, but live reproduction remains
unperformed. Its exact-repeat detector differs from published SBERT, and mock
outputs must never be reported as FIRE accuracy.

## Stockfish and NNUE: what transfers

[Official NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
and [Stockfish search implementation](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp).
Stockfish iteratively deepens principal-variation alpha-beta search; transposition
entries, history-based move ordering, aspiration windows, reductions, futility and
probabilistic pruning exploit a game with legal transitions, alternating opponents,
fixed rules and a scored outcome. Oracle lacks those minimax bounds and a known
complete action/outcome space. A tree-shaped controller does not justify alpha-beta.

Reusable ideas: bounded iterative refinement, action ordering, complete state keys
and carefully invalidated cached evaluations. A chess transposition is exact only
for the relevant state/rule context; Oracle equivalence additionally needs entities,
logic, time, permissions, provenance, corpus and model/policy version.

NNUE accumulates sparse feature-transformer weight columns; many moves change few
active features. Some feature changes require accumulator refresh. Efficient shallow
layers, clipped activations, quantized inference and CPU layouts matter only after
sparse feature construction is cheap. Evidence encoders can be dense and expensive;
source-group merges/retractions may change many embeddings and relation terms.
Training from game positions/evaluations and validation through playing strength do
not transfer into factual labels or calibration. Measure total CPU query cost,
not just accumulator microseconds. No GPL code or weights incorporated.

## DeepMind planning and symbolic verification

[AlphaZero](https://arxiv.org/abs/1712.01815), algorithm and Methods: policy/value
network guides MCTS; training combines outcome squared loss, search-policy cross
entropy and regularization from self-play. Its enormous self-play infrastructure
and reliable game outcomes are absent for web research. Supervised ranking of
logged retrieval actions is the first comparison, not reinforcement learning.

[MuZero](https://arxiv.org/abs/1911.08265), §§2–4 and supplementary ablations:
representation, recurrent dynamics and prediction learn policy/value/reward relevant
to planning. Learned dynamics do not establish truth about unobserved documents.
Oracle has neither a validated retrieval simulator nor enough outcome trajectories;
adding a world model would introduce hallucinated planning states and training cost.

[AlphaGeometry2](https://www.jmlr.org/papers/v26/25-1654.html), §§2–6 and ablations:
expanded formal language, optimized symbolic engine, Gemini-based proposals and
Shared Knowledge Ensemble of Search Trees. The paper reports that larger search
settings do not monotonically improve solving; its single-tree ablation selects
beam size 128, depth four, 32 samples. Formal proved predicates can be shared safely
under their assumptions. Web evidence cannot inherit that status merely because
multiple trees or pages repeat it. Transfer grounded arithmetic/proof traces and
versioned dependency sharing, not theorem-level guarantees about extracted facts.
The original AlphaGeometry implementation/paper merits a separate deeper review;
AG2's account is not treated as a substitute for that remaining work.

## Evidence aggregation

[Deep Sets](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html),
Theorem 2 and experiments: sum-transform aggregation provides a useful invariant
architecture under stated domain/representation conditions; the paper distinguishes
countable and uncountable cases. Finite-dimensional practical networks do not gain
unlimited expressivity or independence awareness from permutation invariance.

[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html), attention blocks
and experiments: pair interactions can improve some set tasks; dense attention is
quadratic in evidence count, inducing-point attention reduces dependence on set
size for fixed inducing count. Its gains on point-cloud/statistical tasks do not
prove factual aggregation quality. Compare grouped logistic, MLP and Deep Sets
before attention. Sparse relation-aware aggregation needs reliable dependency edges;
missing edges create false independence, while dense graphs increase cost. Detailed
GNN implementation/ablation review is still pending.

## Datasets and external validity

[FEVER](https://aclanthology.org/N18-1074/): Wikipedia-derived claims and evidence
labels, including not-enough-information; evidence-aware evaluation matters in
addition to label accuracy. Synthetic mutations and snapshot scope can create
annotation shortcuts and differ from live agent questions.

[FEVEROUS](https://arxiv.org/abs/2106.05707): text and table evidence, cell contexts
and numerical/multi-source reasoning. Its baseline combines retrieval/cell extraction
with NLI and reports 18% joint evidence/verdict success; arithmetic alone cannot
repair omitted headers, units or incorrect table extraction.

[HoVer](https://aclanthology.org/2020.findings-emnlp.309/): up to four Wikipedia
articles and transformed HotpotQA claims; more hops make retrieval harder. Its
NOT-SUPPORTED label is not equivalent to Oracle NO. Article/claim family leakage
and gold-document versus retrieved-document settings must be separated.

[AVeriTeC official page](https://fever.ai/dataset/averitec.html) links a noncommercial
license. It is excluded pending rights review, including evaluation connected to
commercial development. No benchmark dataset has been bundled or used for training.
Temporal cutoffs, stale pages, publication times versus observation times, source
families and closed-corpus/live-web differences remain explicit evaluation axes.

## Calibration, selective prediction and stopping

[Guo et al.](https://proceedings.mlr.press/v70/guo17a.html), §§3–5: held-out
temperature scaling minimizes NLL while retaining class argmax; isotonic methods
can fit nonparametric mappings but need enough held-out data. Brier score is a
proper scoring rule. ECE depends on bins, sample size and selected population;
low aggregate ECE does not guarantee correctness on unfamiliar questions.

[Geifman & El-Yaniv](https://proceedings.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html),
Algorithm 1/Theorem 3.2: selective classification trades coverage for risk under
sampling assumptions and accounts for threshold selection. [Bates et al.](https://arxiv.org/abs/2101.02703)
uses held-out calibration and upper risk bounds for prediction sets. Neither is a
license to interpret each NLI softmax as an individual truth probability.

Current Oracle calibration is fixed-bin empirical correctness with a conservative
one-sided Wilson approximation, bound to policy/model versions and expiry. It is
**not** an implementation of those risk-control guarantees, conformal prediction,
anytime-valid sequential testing or distribution-shift robustness. Repeated
adaptive stops require calibration of the final policy or a separately proved
sequential procedure; the latter is still pending research.

## Architecture selection matrix

Scores are subjective engineering priors, **not measured results**. Five is
preferable (including lower cost/difficulty/risk); one is least favorable. They
assume commodity CPU, BYO retrieval and no licensed training corpus yet.

| Criterion | A deterministic | B FIRE-style | C two evaluators | D NNUE-inspired | E graph/policy |
|---|---:|---:|---:|---:|---:|
| Mathematical auditability | 5 | 2 | 3 | 3 | 2 |
| Implementation simplicity | 5 | 4 | 3 | 2 | 1 |
| Eligible training-data availability | 5 | 4 | 2 | 2 | 1 |
| Inference economy | 5 | 2 | 4 | 4 | 2 |
| Search economy, unmeasured prior | 2 | 3 | 3 | 3 | 3 |
| Network latency, unmeasured prior | 3 | 2 | 3 | 3 | 2 |
| Memory economy | 5 | 3 | 3 | 4 | 1 |
| Factual error auditability | 5 | 2 | 3 | 3 | 2 |
| Calibration tractability | 3 | 2 | 3 | 3 | 2 |
| CPU suitability | 5 | 2 | 4 | 5 | 3 |
| Incremental-update potential | 4 | 2 | 3 | 5 | 4 |
| Offline/self-host suitability | 5 | 2 | 5 | 5 | 4 |
| Familiar/unfamiliar semantic coverage | 1 | 4 | 3 | 3 | 3 |
| Maintenance simplicity | 5 | 3 | 3 | 2 | 1 |

A is simplest to inspect but has deliberately narrow coverage; its low model cost
cannot be compared fairly without accounting for abstention and search. B offers
broad language handling with provider/token costs and weak native provenance.
C adds supervised semantic relationships while preserving the evidence engine.
D's favorable CPU/update scores are conditional on sparsity and stable group
features; total CPU gains remain unknown. E needs graph labels and trajectories
that are not yet available. All candidates share network costs; no inference
optimization proves search savings. A Gemini hybrid may improve coverage while
adding vendor cost, privacy/retention checks and a new calibration population.

Select A as the durable baseline, C as first challenger and E as high-risk research.
Scores are not added into a fictitious aggregate benchmark. D and E must beat the
simpler deployed policy under the preregistered adoption gates.
