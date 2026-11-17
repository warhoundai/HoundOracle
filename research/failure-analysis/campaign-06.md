# Campaign 06: balanced gains did not transfer

The one-epoch MiniLM encoder and logistic aggregator improved old validation accuracy from52.92% to75.42% at the .80 evaluation policy. After fixing the .85 routing threshold using old validation, balanced heldout correct-definitive coverage rose from46/300 to117/300. Nevertheless the strict-page heldout selected error was13/32 (40.625%), above the20% ceiling, versus2/12 for the original model. Small accepted samples make the original model's risk uncertain too; it is preserved rather than newly certified for this shifted population.

Inspection of retired examples suggests missing entity/source grounding, negation and numerical scope errors. Passage text may be about a similarly named work or use unresolved pronouns; a text-only NLI model can confidently evaluate the wrong entity. Correct NLI entailment is also not equivalent to sufficient real-world truth evidence. Some gold NEI rows appear to contain superficially relevant material: these require independent adjudication, not post-hoc relabeling to inflate model scores.

An agreement guard with the old pretrained NLI did not solve the problem economically. Calibrating against the worse retired regime reduced definitive coverage to12.5% on balanced and1% on strict examples. This candidate was rejected before successor testing.

The next isolated architectural change restricts the new model to insufficiency detection after the original verifier abstains. It cannot add YES/NO answers. This preserves original definitive outputs, but may still withhold an answer Gemini could establish, and adds a second encoder pass to fallback queries. Therefore local skipped-call counts do not establish end-to-end quality or latency improvements. Live paired Gemini evaluation is necessary before promotion.

Profile measurements put97.79% of local time in the existing native ONNX runtime call. C++ orchestration would address less than~.626ms/query on this workload and would not reduce Gemini or network latency. Source-aware evidence compilation, conservative claim-type routing and independently labeled retrieval evaluations deserve priority.
