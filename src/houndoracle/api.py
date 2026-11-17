"""Authenticated, single-tenant REST facade. Deploy separate stores per tenant."""

import secrets
from contextlib import asynccontextmanager

from pydantic import Field

from .models import Strict, VerifyRequest


class SearchRequest(Strict):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(10, ge=1, le=100)
    allow_network: bool = False
    max_requests: int = Field(4, ge=1, le=4)
    max_seconds: float = Field(20, gt=0, le=120)
    max_cost_usd: float = Field(0, ge=0, le=1)
    prefer_local: bool = True


class CaptchaRequest(Strict):
    image_base64: str = Field(min_length=1, max_length=80000)
    backend: str = "tesseract"
    alphabet: str = Field(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        min_length=1,
        max_length=62,
    )
    challenge_type: str = Field("text", max_length=32)
    expected_length: int | None = Field(None, ge=1, le=32)


class GoogleRecaptchaRequest(Strict):
    target_url: str = Field(min_length=1, max_length=4096)
    allow_network: bool = False
    max_seconds: float = Field(90, gt=0, le=120)


class ReleaseRecordRequest(Strict):
    question: str = Field(min_length=1, max_length=4000)
    allow_network: bool = False


def create_app(
    oracle, *, api_key, enable_captcha=False, recaptcha_config=None, release_records=None
):
    if not api_key or len(api_key) < 24:
        raise ValueError("API key must contain at least 24 characters")
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from .jobs import Jobs

    jobs = Jobs(oracle)

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.close()

    app = FastAPI(
        title="HoundOracle",
        version="0.1.0.dev1",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.jobs = jobs

    @app.middleware("http")
    async def authorize(request: Request, call_next):
        if request.url.path != "/health":
            supplied = request.headers.get("authorization", "")
            if not secrets.compare_digest(supplied.encode(), ("Bearer " + api_key).encode()):
                return JSONResponse({"error": "UNAUTHORIZED"}, status_code=401)
        if request.method == "POST":
            body = bytearray()
            async for part in request.stream():
                body.extend(part)
                if len(body) > 65536:
                    return JSONResponse({"error": "BODY_TOO_LARGE"}, status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(PermissionError)
    async def forbidden_network(request: Request, exc):
        return JSONResponse({"error": "LOCAL_ONLY_NETWORK_DISABLED"}, status_code=403)

    @app.exception_handler(KeyError)
    async def not_found(request: Request, exc):
        return JSONResponse({"error": "NOT_FOUND"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request: Request, exc):
        return JSONResponse({"error": "INVALID_OPERATION"}, status_code=422)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "release": "development",
            "calibrated": oracle.calibration is not None
            and getattr(oracle.calibration, "qualification_status", "audited") == "audited",
            "calibration_status": getattr(
                oracle.calibration,
                "qualification_status",
                "legacy_configured" if oracle.calibration is not None else "absent",
            ),
            "general_verification_enabled": oracle.general_verifier is not None,
        }

    if enable_captcha:

        @app.post("/v1/captcha/solve")
        def captcha_solve(request: CaptchaRequest):
            from .captcha import solve_encoded_captcha

            return solve_encoded_captcha(
                request.image_base64,
                backend=request.backend,
                expected_length=request.expected_length,
                alphabet=request.alphabet,
                challenge_type=request.challenge_type,
            )

    if recaptcha_config:

        @app.post("/v1/captcha/google")
        def google_recaptcha(request: GoogleRecaptchaRequest):
            from .recaptcha import solve_google_recaptcha

            return solve_google_recaptcha(
                request.target_url,
                allow_network=request.allow_network,
                max_seconds=request.max_seconds,
                **recaptcha_config,
            )

    if release_records is not None:

        @app.post("/v1/tools/release-records")
        def release_record_verify(request: ReleaseRecordRequest):
            if request.allow_network and oracle.local_only:
                raise PermissionError("LOCAL_ONLY_NETWORK_DISABLED")
            return release_records.verify(request.question, allow_network=request.allow_network)

    @app.post("/v1/verify")
    def verify(request: VerifyRequest):
        options = request.model_dump(exclude={"question", "claim"})
        return oracle.verify(request.question, claim=request.claim, **options).model_dump(
            mode="json"
        )

    @app.post("/v1/search")
    def search(request: SearchRequest):
        if request.allow_network:
            return oracle.web_search(
                request.query,
                min(request.limit, 10),
                **request.model_dump(exclude={"query", "limit"}),
            )
        return {"results": oracle.search(request.query, request.limit), "verified": False}

    @app.get("/v1/explanations/{result_id}")
    def explain(result_id: str):
        return oracle.explain(result_id)

    @app.post("/v1/jobs", status_code=202)
    def submit(request: VerifyRequest):
        return jobs.submit(request)

    @app.get("/v1/jobs/{job_id}")
    def get(job_id: str):
        return jobs.get(job_id)

    @app.post("/v1/jobs/{job_id}/{action}")
    def transition(job_id: str, action: str):
        return jobs.transition(job_id, action)

    @app.get("/v1/openapi.json")
    def openapi():
        return app.openapi()

    default_openapi = app.openapi

    def secured_openapi():
        schema = default_openapi()
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["InstanceKey"] = {
            "type": "http",
            "scheme": "bearer",
        }
        for path, methods in schema.get("paths", {}).items():
            if path != "/health":
                for operation in methods.values():
                    operation["security"] = [{"InstanceKey": []}]
        return schema

    app.openapi = secured_openapi
    return app
