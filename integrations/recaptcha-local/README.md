# Experimental local Google reCAPTCHA v2 helper

**Not qualified for reliable use:** live EC2 attempts reached real image challenges but did not confirm completion within120seconds. This integration is available for research; do not advertise Google as fixed.

This directory is **AGPL-3.0-only**, separately licensed from HoundOracle's Apache-2.0 Python core. It depends on `@conghuy113/recaptcha-solver`0.5.1 and its local ONNX classification/detection models. The lockfile overrides its original runtime pins with ONNX Runtime1.31.1, sharp0.35.5, ws8.21.0 and Puppeteer25.17.0 after dependency auditing. Keep this LICENSE and upstream notices when distributing it; modifications/distribution/network use must meet the applicable AGPL terms. Core licensing is not a claim that these dependencies are Apache-licensed. See UPSTREAM_MODEL_LICENSES.md and UPSTREAM_DATASET_PROVENANCE.md for model licensing and unresolved image-level provenance limitations. No training dataset or weights are vendored here.

## Setup

Node22.12+ and a maintained local Chromium installation are required. Linux x64 is the tested path; other platforms are unqualified. Install only the pinned dependency graph, then explicitly prepare models:

```bash
cd integrations/recaptcha-local
RECAPTCHA_SOLVER_SKIP_MODEL_DOWNLOAD=1 npm ci --ignore-scripts
# Install-time model preparation; may download about350MB of hash-pinned model files.
RECAPTCHA_SOLVER_MODEL_DIR=/absolute/private/recaptcha-models \
  node node_modules/@conghuy113/recaptcha-solver/dist/postinstall.js
```

Use the same absolute model directory at runtime. Missing/wrong-size/wrong-hash models fail before browser interaction. Runtime fetches for model downloads are blocked. Inference is CPU-local; Google still receives its own widget/browser network traffic. There is no commercial solver fallback, cloud inference, proxy rotation or fingerprint-spoofing layer.

Start a **dedicated automation Chrome profile** with debugging on loopback, then open the exact Google challenge URL in it:

```bash
chromium --remote-debugging-address=127.0.0.1 --remote-debugging-port=9222 \
  --user-data-dir=/absolute/private/oracle-google-profile
```

Never expose CDP to the network or point this helper at a general-purpose personal browser. CDP grants broad browser authority. This helper is a different, opt-in browser boundary from HoundBrowser's exact-origin read-only service: it does not weaken HoundBrowser's policy or expose that service's CDP endpoint.

From Oracle's repository:

```bash
houndoracle recaptcha-solve 'https://www.google.com/recaptcha/api2/demo' \
  --helper-dir "$PWD/integrations/recaptcha-local" \
  --model-dir /absolute/private/recaptcha-models --cdp-port 9222 \
  --max-seconds 90 --allow-network
```

The target tab must already exist with that **exact URL**. The helper accepts only`https://www.google.com`; it never selects a different tab by substring, launches a replacement session, submits other forms, or returns cookie/response-token contents to Oracle. It uses the library's local3×3/4×4 recognizers and checkbox/Verify interactions in that existing context. The browser remains open for the agent to continue after the helper disconnects. v3/Enterprise risk scoring and non-Google-origin widgets are not supported by this adapter.

## Agent interfaces

```bash
houndoracle --recaptcha-helper-dir "$PWD/integrations/recaptcha-local" \
  --recaptcha-model-dir /absolute/private/recaptcha-models mcp
```

MCP adds`recaptcha_solve(target_url, allow_network, max_seconds)`. The same configuration with`serve` adds authenticated`POST /v1/captcha/google`. The request must explicitly set`allow_network:true`. The operator fixes helper/model paths and the loopback CDP port; API callers cannot choose an arbitrary executable or remote browser. Cloud/service keys are stripped from the helper environment. The process wall clock is capped at120seconds, and a timeout reports`TIMEOUT_OUTCOME_UNKNOWN`; do not replay uncertain interactions automatically.

`WIDGET_SOLVED` requires upstream confirmation by a solved widget/new token or navigation evidence. A token alone is unverified. `image_challenge_solved` distinguishes actual image rounds from a checkbox that needed no image challenge. **Application-server acceptance is not inferred** (`server_accepted:null`). Cookies/tokens stay in the existing local browser. No challenge result is factual evidence.

Solving a widget in this browser does not clear SearXNG's separate HTTP session or guarantee Google Search access. Continue in the same browser context. Do not copy tokens to a different session or describe a successful demo as a reliable Google search API.

## Qualification

Node contract tests run without model downloads: `node --test bridge.test.mjs`. Python tests cover explicit permission, exact target checks and secret stripping. Live attempt/results and remaining limitations are recorded under`research/experiments/recaptcha-26/`. One demo attempt cannot establish a success rate or universal Google compatibility. Google may still refuse automated clients.
