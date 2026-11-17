# Three-class calibration adjustment — rejected

A single fixed regularized multinomial logistic map adjusted all3relation logits.
The existing1000calibration-fit components were split by hash into500class-map fit
and500correctness-map fit groups. Neither overlapped the207development queries.
No encoder gradients, hyperparameter sweep, fresh risk or locked scoring occurred.

Raw accuracy174/207(84.06%) versus176/207(85.02%). At the provisional80%policy both
routes produced103correct definitive answers out of207with109accepted. No material
benefit; reject. The classifier map is not a factual correctness estimate, and the
separate fitted correctness map remains unqualified. JSONstate validates class
order and finite tensor shapes; no pickle or executable model artifact is used.
