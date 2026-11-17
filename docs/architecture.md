# Architecture and mathematical specification

Status: deterministic development baseline. Architecture C's NLI model is an
optional research interface, not a qualified production evaluator. Exactly two
primary evaluator interfaces exist; the default first evaluator uses exact
controlled-language extraction and the second is grouped logistic regression.
No chess, theorem-proving or neural code has been incorporated from research repos.

## State, semantics and data flow

Let `S_t=(Q,G_t,U_t,B_t,T_t)`: Q is the original question plus a typed proposition;
G is versioned documents, spans and provenance edges; U is unresolved atomic
propositions; B is remaining request/money reservations; T is remaining wall time.
An action issues an atom-specific primary-record query, fetches evidence, or
revalidates a version. Observations are bounded source material, an explicit
no-result, or an operational failure. `Update` adds evidence without granting it
curator trust, recomputes dependency groups, and deducts reservations/time.

Terminal states: qualified YES/NO; unsupported or ambiguous semantics; insufficient
or conflicting evidence; budget/time exhaustion; exhausted action queue; retrieval
failure; cancellation; pause. Only clearly non-propositional input is INVALID.
Operational errors never provide negative evidence. Failed retrieval stops the
current baseline search; recovery requires an explicit new request/job action.

The current priority queue orders atomic primary-record actions deterministically.
It does not implement learned best-first utility, contradiction-specific actions,
beam search, UCB or MCTS. These are comparison candidates, not shipped features.
Searches are not webpages-as-tree-nodes: each action updates a verification state.
The queue is bounded by atomic count, unique queries, request cap, time and money.

Canonicalization preserves entity spelling, years, numeric operators, units,
Boolean nesting and question polarity. Only double negation and commutative
ordering of explicitly typed AND/OR are normalized. Similarity cannot establish
logical equivalence. Supplied typed claims must match a compiled question or use
`verify_claim`, which generates an exact structured question. No natural-language
paraphrase model may silently replace this check.

Grounded atomic truth uses reviewed primary-source records and rejects conflicting
reviewed records. Three-valued strong Kleene operators propagate unknown under
negation. The current composite decision policy additionally requires every atom
resolved, even when a partial Boolean proof would suffice. This is conservative
coverage loss, not a complete logical theorem prover. Time-dependent quantities
without an explicit temporal claim representation remain out of scope.

## Evaluators, losses and costs

Semantic research model: `z_i=f_theta(Q,e_i)`, `p_i=softmax(Wz_i+b)`. Use supervised
entailment/contradiction/neutral cross entropy and independently measure entity,
date, number, unit and exact-span extraction. NLI entailment is a relationship
between texts, not a calibrated probability of factual truth. LocalNLI refuses
implicit downloads, remote code and unsafe pickle weights. It is not connected to
the production decision gate.

Document hashes, declared common origins and upstream edges create transitive
source groups. Those are conservative known dependencies, not proof that remaining
groups are independent. The baseline uses three log-count features: support-only,
refutation-only and internally conflicting reviewed groups. It fits binary
cross-entropy plus L2 regularization using deterministic gradient descent. Default
weights are disclosed heuristics. Symbolic truth selects the verdict; the minimum
per-atom score feeds population-specific final-policy calibration.

Deep Sets candidate: `h=sum_g phi(z_g)` and `p=softmax(rho(h))`; O(Gd) aggregation
work and O(d) running state after group embeddings are computed. Relation extension:
`h=sum_g phi(z_g)+sum_(g,j in E) psi(z_g,z_j,r_gj)`, with relation type, citation
lineage, content duplication, version and contradiction features. Sparse work is
O((G+|E|)d); dense pairwise work is O(G^2 d). Neural encoders and graph discovery
costs must be counted separately. Group merger or encoder/model changes can require
full recomputation; bounded-delta updates are not generally assured.

NNUE hypothesis: `h_next=h-u_old+u_new+Delta_relations`. This is exact only when
embeddings of unaffected groups are unchanged and every modified relation is
removed/re-added. Measure update/full-recompute equivalence, actual feature sparsity,
CPU time, memory, quantization error and total query time. No incremental neural
implementation is currently promoted; current grouping fully recomputes.

Search utility hypothesis:
`U(a)=R_stop(S)-E[R_stop(S_next)|S,a]-lambda_c*C(a)-lambda_t*L(a)`.
Use USD/query for C and seconds for L, with lambdas in loss/USD and loss/second.
Estimate action outcomes from held-out trajectories; unknown future documents are
not observable to the controller. This is a myopic approximation, not an optimality
proof. Current queue priorities do not claim to estimate this utility.

`C_total=C_search+C_fetch+C_encode+C_evaluate+C_host+C_fallback+C_sync`.
For hit rate h, `E[C]=h*C_hit+(1-h)*C_miss+C_sync`. If C_miss>C_hit, caching beats
baseline C_L when `h>(C_miss+C_sync-C_L)/(C_miss-C_hit)`. This does not imply a
feasible h in [0,1]. Host utilization, provider rates, model sizes, sync rates,
training, engineering and maintenance remain unmeasured. Latency follows the
critical path plus queueing; sum of request latencies is inappropriate for parallel
retrieval. The current engine retrieves sequentially and reports local wall time.

## Evidence lifecycle and safety boundaries

SQLite stores document versions, exact text, hashes, source metadata, permissions,
provenance edges, answer dependencies and durable job records. BM25/FTS5 supports
local discovery; verification scans a bounded corpus to avoid overlooking a
contradiction merely because top-k ranking omitted it. Corpus limits: 1,000 active
documents and 5 MB of active serialized evidence. Larger corpora abstain until a
qualified evidence-completeness strategy is implemented.

Revalidation uses ETag/Last-Modified. A 304 renews freshness and invalidates existing
answers for recomputation; changed content loses review/sharing permission. Source
failure revokes the version. Corrections invalidate downstream versions recursively.
New evidence conservatively invalidates all cached conclusions to catch new
contradictions. An ancestor's expiration makes descendants ineligible. This favors
correctness over cache hit rate and is not claimed to scale to a central index.

S3 stores immutable content-addressed blobs and signed, expiring manifests in an
operator-supplied private versioned bucket. Conditional writes deduplicate blobs.
Upload/download require explicit authorization. Only reviewed, permissioned public
content with an allowed license is eligible. Local imports reset trust by default.
S3 storage and signatures do not verify factual truth. There is no central public
contribution API or DynamoDB/Lambda deployment yet.

HTTP retrieval validates each redirect, blocks non-public DNS answers and pins the
socket to the checked address. Remote adapters require TLS; private endpoints need
an explicit operator allowlist. Retrieved content cannot invoke tools or grant
permissions. Provider failures are redacted. No telemetry or answer-to-evidence
feedback path exists. One REST instance/store is one tenant; multi-tenant deployment
has not been implemented or qualified.

Per-query request/cost reservations are atomic and made before provider calls.
Reported reserved cost is an upper reservation, not an invoice; actual cost is null
until metered. A HoundSearch adapter upper bound must include downstream provider
work and be independently enforced by HoundSearch. Time checks are cooperative;
OS DNS resolution, arbitrary provider code, model inference and in-flight requests
cannot be universally interrupted at the deadline. Do not advertise a hard runtime
or cloud-wide spending guarantee.

Durable jobs support status, cooperative pause/cancel and budget-preserving resume
within one process. Restarted or uncertain jobs pause for reconciliation rather
than replay paid calls. Seen-query/frontier checkpoints are not yet persisted;
resume may repeat retrieval using remaining budget. Central budgets and fleet-wide
reservations remain a deployment gate.
