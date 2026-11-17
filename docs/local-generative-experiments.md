# Local open-weight adjudication experiments

`houndoracle.experiments.generative.LocalGenerativeVerifier` accepts an already
loaded local inference client. It does not download weights, call a hosted model,
reuse previous answers as evidence, or qualify a production verdict. Candidate
outputs have no calibrated correctness score and remain explicitly unqualified.

The measured CPU candidate uses Apache2 Qwen weights in GGUF form and the MIT
`llama-cpp-python==0.3.16` runtime. Weights and their upstream licenses remain
separate from this repository; exact revisions, sizes and SHA256 hashes are in the
experiment protocols. Verify the complete file hash before loading. For an
installed, verified model, inference needs no network or central Oracle account.

```python
from llama_cpp import Llama
from houndoracle.experiments.generative import LocalGenerativeVerifier

# Install the separately licensed optional runtime and verify model.gguf first.
client = Llama(model_path="/models/model.gguf", n_ctx=4096,
               n_threads=8, n_gpu_layers=0, verbose=False)
verifier = LocalGenerativeVerifier(client, config={"checkpoint_sha256": "verified hash"})
result = verifier.predict("The exact proposition", [
    {"url": "https://example.org/record", "text": "A permitted evidence passage"}
])
# result["candidate"] is an experimental candidate, not an authorized YES/NO.
```

The prompt preserves the original proposition as data and distinguishes
contradiction from missing support. JSON-schema-constrained generation identifies
zero-based evidence indices. Invalid output or uncited definitive answers are
recorded as failures; an empty corpus returns MAYBE without invoking the model.
An in-range source index proves neither independence nor entailment. Prompt
instructions alone do not establish resistance to adversarial evidence.

For Ubuntu CPU builds with BLAS, prerequisites include a compiler, CMake,
`pkg-config`, Python development headers and OpenBLAS development files. The first
EC2 attempt failed for missing `pkg-config`; it produced no accuracy observations.
Its failure is retained, followed by one bounded repaired attempt.

The4B candidate scored32/48correct on the frozen balanced development screen,
versus35/48for adapted DeBERTa; reject it as a replacement. The larger-model
controlled follow-up is recorded separately in campaign34. Do not interpret a
larger model or a valid JSON response as improved factual reliability.
