# Experimental Google reCAPTCHA v2 integration

Oracle now has an opt-in local helper for Google's reCAPTCHA v2 checkbox and3×3/4×4 image challenges. It uses a pinned open-source solver with CPU-local ONNX models and an existing dedicated Chromium session. It is separate from the text-image OCR tool.

[Install and operate the helper](../integrations/recaptcha-local/README.md). It is separately **AGPL-3.0-only**, including its model license obligations; it is not included in the Apache-licensed Python wheel. Images are fetched from narrowly allowed Google reCAPTCHA endpoints for local inference. There is no paid solver or hosted inference fallback. Model setup downloads hash-pinned weights once; runtime model downloads are blocked.

The native CLI is `houndoracle recaptcha-solve`. Operator configuration enables the `recaptcha_solve` MCP tool or authenticated`POST /v1/captcha/google`. The exact target tab must already exist at`https://www.google.com`. Chrome's CDP endpoint stays on loopback. The helper preserves that browser's state, strips cloud/service credentials from its subprocess environment and omits cookies/tokens from its response. A solved widget is not asserted to be application-server acceptance. Timeouts are uncertain outcomes and must not be replayed automatically.

This does not clear a separate SearXNG HTTP session, provide a general Google search API, support reCAPTCHA v3/Enterprise risk scoring, or qualify Oracle's factual accuracy. Continue the agent workflow in the same browser context. HoundBrowser's independent origin/method restrictions are not silently changed to accommodate this helper.

Live EC2 attempts reached real Google3×3 challenges but did not confirm completion within120seconds. All five temporary workers were terminated. The live EC2 qualification campaign and failures are recorded in [recaptcha26](../research/experiments/recaptcha-26/README.md). The helper remains experimental until reliable completion is demonstrated; a unit-test pass or a downloaded model is insufficient evidence.
