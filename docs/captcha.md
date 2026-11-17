# Experimental local text-CAPTCHA solver

Oracle now includes automated OCR for **text-image CAPTCHAs**. It runs on the user's CPU using open-source Tesseract (default) or `ddddocr`1.6.1 (experimental ONNX backend). Images are not uploaded to a paid solver, Google, Gemini or a central Oracle service. Install dependencies/models once; inference works locally without internet.

This does **not** solve reCAPTCHA/hCaptcha image grids, audio puzzles, sliders, proof-of-work or browser-attestation challenges. A separate [experimental Google reCAPTCHA v2 helper](google-recaptcha.md) now exists; its live qualification is separate from these OCR results. Google's earlier access-denied/JavaScript-gate failure is not resolved by text OCR. Do not label this a general Google CAPTCHA bypass.

## Install and run

```bash
# Debian/Ubuntu; use your OS package manager on other systems.
sudo apt-get install tesseract-ocr tesseract-ocr-eng
uv sync --extra captcha --extra api --extra mcp
uv run houndoracle captcha-solve /path/to/captcha.png --length 5
# Alternate local pretrained recognizer:
uv run houndoracle captcha-solve /path/to/captcha.png --backend ddddocr --length 5
```

If a site specifies a digits-only or uppercase-only field, declare that constraint rather than guessing from the image:

```bash
uv run houndoracle captcha-solve /path/to/captcha.png --alphabet 0123456789 --length 5
```

The result contains `status`, `candidate`, `backend`, `latency_seconds`, and `server_accepted:false`. `CANDIDATE` means an unverified recognition result. The calling agent can use its already configured browser/form tools to submit a candidate and check the site's actual response. Recognition alone is not proof of acceptance; no form is submitted by the OCR module. Wrong length, malformed images, missing dependencies and timeouts have separate statuses. No OCR score is exposed as calibrated correctness.

Python:

```python
from pathlib import Path
from houndoracle.captcha import solve_text_captcha
result = solve_text_captcha(Path('captcha.png').read_bytes(), expected_length=5)
```

## Agent interfaces

```bash
uv run houndoracle --captcha mcp
# Local authenticated REST endpoint:
export HOUNDORACLE_API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
uv run houndoracle --captcha serve
```

MCP adds `captcha_solve(image_base64, backend, expected_length, alphabet, challenge_type)`. REST adds authenticated `POST /v1/captcha/solve` with the same JSON fields. Both are opt-in so existing tools remain unchanged. REST retains its64KB whole-request body limit; CLI/MCP accept images up to1MB. Setting `challenge_type` to anything other than`text` returns `UNSUPPORTED_CHALLENGE` without starting inference. A screenshot of a whole browser challenge is not equivalent to a cropped text-CAPTCHA image.

An internet-enabled container profile includes both local backends:

```bash
docker compose -f deployment/selfhost/compose.web.yaml \
  -f deployment/selfhost/compose.captcha.yaml up --build -d
```

Compose configuration was validated; this new image was not fully built in the disk-constrained development workspace. The native OCR backends, CLI, authenticated API and MCP contracts were exercised. Existing EC2 search/deployment tests do not count as testing this new image.

## Bounds and privacy

The recognizer accepts bounded single-frame PNG/JPEG bytes, limits decoded dimensions/pixels/aspect ratio, and never accepts a URL or arbitrary command as image input. Inference runs in a short-lived subprocess with a wall-clock timeout, CPU limit on supported systems, stripped service/cloud credentials, one-thread numeric settings and Python socket calls disabled. POSIX timeout cleanup kills the worker process group, including a Tesseract child. This is resource containment, not a complete adversarial native-code sandbox. Untrusted model paths and executable model code are not accepted.

`ddddocr` comes from https://github.com/sml2h3/ddddocr and declares MIT for its repository/distribution; installed model hashes and wheel hash are recorded in [upstream.json](../research/experiments/captcha-25/upstream.json). Its model training-data provenance has not been independently established. No weights or upstream code are vendored in Oracle. Tesseract declares Apache-2.0: https://github.com/tesseract-ocr/tesseract. Installed dependencies/data retain their own notices.

## Measured limits

On40 locally generated development images (digits, uppercase, mixed case, noise and rotation), Tesseract matched26/40 exactly (65%); adjusted`ddddocr` matched25/40 (62.5%). Tesseract's measured process-inclusive P50/P95 were approximately0.305/0.424seconds;`ddddocr`0.375/0.541seconds on this machine. The original`ddddocr`configuration matched15/40; retaining OCR characters before projecting to a caller-declared single-case alphabet improved those development results. Mixed-case fields are never case-folded.

These small synthetic cases are not an independent real-world CAPTCHA benchmark. The same cases informed the normalizer/default selection. They establish neither99% accuracy nor successful Google challenge completion. Failures and per-example records are retained in [the experiment](../research/experiments/captcha-25/README.md). No change to Oracle's factual-verification calibration or best model is implied.
