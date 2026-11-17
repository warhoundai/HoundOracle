# Experimental local Google reCAPTCHA v2 integration

**Qualification gate not passed: no completed Google image challenge was confirmed.** The implementation remains an opt-in research helper, not a default retrieval dependency or a reliable Google Search access claim. No public release.

User requested automated Google CAPTCHA support. Investigated original open-source implementations, README, package/runtime source, model manifests, licenses, provenance notes and public registry metadata. Selected conghuy113/solveReCaptcha-Typescript0.5.1 for a separate AGPL-3.0-only helper with local3×3/4×4 inference and no hosted solver fallback. A different MIT-labelled candidate explicitly included cloud fallback/stealth and was rejected for this profile. See upstream.json. Model provenance limitations remain recorded; no training dataset or model weights are vendored in Oracle.

## Implementation and acceptance

Python's Apache-licensed CLI/API/MCP communicate with the independently licensed Node helper through bounded JSON stdio. Operators choose the helper/model directories and a loopback CDP port. Requests need explicit network permission and an exact existing `https://www.google.com` tab. The browser remains owned by its caller. Cookies/response tokens are not returned or inserted into evidence. A token alone cannot satisfy `WIDGET_SOLVED`; application-server acceptance remains unknown. A timeout is `TIMEOUT_OUTCOME_UNKNOWN`, with no automatic retry. This cannot clear SearXNG's separate HTTP session.

Model files are prevalidated by size/SHA256. Runtime model download is blocked. Image acquisition is restricted to HTTPS Google/reCAPTCHA v2 payload endpoints, no redirects, at most32 image requests,2MB per image and16MB aggregate. The browser itself makes additional widget traffic; these bounds are not a complete count of Chrome network requests. Missing/failed native runtime is not a successful solve. No proxy rotation, fingerprint-spoofing options or token replay were added.

187 Python tests pass on3.11/3.12,6 Node tests pass, lint and wheel build pass. Node tests cover token-only rejection, secret omission, approved target/transport boundaries and missing models. Python tests cover permission checks, operator-controlled paths/port, credential stripping and authenticated optional interfaces. These are not live-solving success measurements.

## Dependency correction

The initial graph (solver0.5.1/Puppeteer24.31.0) reported14 high-severity audit entries, including dependency propagation. It is not the recommended graph. Pinned Puppeteer25.17.0 and overrides ONNX Runtime1.31.1, sharp0.35.5 and ws8.21.0 yielded zero known npm audit vulnerabilities at inspection. Both audit reports are retained. This does not guarantee absence of vulnerabilities or qualify old browser binaries for production.

## EC2 attempts

All runs used Ubuntu24.04, c7i.2xlarge (8vCPU/16GiB), encrypted24GiB gp3 roots with DeleteOnTermination, IMDSv2, no inbound rules, no IAM instance role,12minute terminate watchdog, and a120second solver cap. Runs1–4 used us-east-1a. A fifth launch there failed for insufficient capacity and created no instance; the fifth worker used the existing default subnet in us-east-1b. Quoted Linux compute price was$0.357/hour. The same$5 campaign reservation covered the bounded runs; actual invoice and all-in monetary cost remain unknown. Compute, publicIPv4, storage, requests and network costs must not be called zero.

| Run | Outcome | Interpretation |
| --- | --- | --- |
| 1 | Demo HTTP200, widget frames present; solver failed in9.35s | Generic error capture was insufficient; not a success |
| 2 | Fire-hydrant3×3 challenge visible; `RUNTIME_DOWNLOAD_BLOCKED`,8.25s | Our guard mistakenly blocked challenge-image retrieval along with model downloads |
| 3 | After narrowly allowing payload images and keeping the browser test event loop responsive, car3×3 challenge remained unchecked at120s | `TIMEOUT_OUTCOME_UNKNOWN`; no confirmed completion |
| 4 | Patched native models executed successfully, then test harness failed on a changed private Puppeteer file path | No Google attempt; no profile output retained; corrected test to use the public package export |
| 5 | Patched runtime/CPU smoke passed; crosswalk3×3 challenge remained unchecked at120s | `TIMEOUT_OUTCOME_UNKNOWN`; no confirmed completion |

These are adaptive debugging runs, not identically configured independent trials; do not calculate or advertise a general success rate from them. No v3/Enterprise or Google Search admission test passed. Public demo interaction is distinct from application-server verification. All raw response tokens/cookies and challenge images were excluded from retained public records.

The patched native zero-tensor smoke returned classification `[1,14]` and detection `[1,84,8400]`. With two intra-op threads, load/inference seconds were respectively0.317/0.445 and0.409/0.923. This is a single CPU interface/profile observation on zeros, not challenge accuracy, batch-grid latency or agent-workflow speed. Images/labels and representative success-rate evaluation remain missing. The killed helper's final image request/byte counters were unavailable on timeout; do not invent them.

## Decision and cleanup

Keep as experimental infrastructure; do not promote as a reliable Google solver. Stop this bounded campaign after the corrected patched-runtime attempt: repeated unchanged live attempts would not supply independent model supervision or demonstrate reliability. Next research needs challenge-level traces, independently reviewed image-task labels/provenance, and a controlled comparison of classifiers/selection policies. Reliable Google Search retrieval remains unresolved, and Oracle's99–100% factual-accuracy target is unaffected/unmet.

All five workers terminated; temporary security group sg-020ee1d9b11b04a5a deleted. No attached test volumes or active HoundOracle workers remained in the cleanup inspection. No IAM role/profile or public application endpoint was created. Private source/log artifacts retain the existing short S3 lifecycle. See cleanup.json, attempt records and source hashes. Existing products and OPS were untouched.
