"""Stdio MCP: process access authorizes access to this single local evidence store."""


def server(oracle, *, enable_captcha=False, recaptcha_config=None, release_records=None):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("HoundOracle")

    @mcp.tool()
    def verify(
        question: str,
        min_confidence: float = 0.8,
        allow_network: bool = False,
        max_cost_usd: float = 0,
    ):
        """Verify from local evidence; MAYBE means insufficient justification."""
        return oracle.verify(
            question,
            min_confidence=min_confidence,
            allow_network=allow_network,
            max_cost_usd=max_cost_usd,
        ).model_dump(mode="json")

    @mcp.tool()
    def search(query: str, limit: int = 10):
        """Discover local evidence; results are not verification verdicts."""
        return {"results": oracle.search(query, limit), "verified": False}

    @mcp.tool()
    def web_search(query: str, max_cost_usd: float = 0, prefer_local: bool = True):
        """Explicitly send this query to configured external search providers; results unverified."""
        return oracle.web_search(
            query, allow_network=True, max_cost_usd=max_cost_usd, prefer_local=prefer_local
        )

    @mcp.tool()
    def explain(result_id: str):
        """Return a stored evidence trace and current validity."""
        return oracle.explain(result_id)

    if release_records is not None:

        @mcp.tool()
        def release_record_verify(question: str, allow_network: bool = False):
            """Experimental exact Node.js release-index proof, not general verification."""
            if allow_network and oracle.local_only:
                raise PermissionError("LOCAL_ONLY_NETWORK_DISABLED")
            return release_records.verify(question, allow_network=allow_network)

    if enable_captcha:

        @mcp.tool()
        def captcha_solve(
            image_base64: str,
            backend: str = "tesseract",
            expected_length: int | None = None,
            alphabet: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            challenge_type: str = "text",
        ):
            """Local text-CAPTCHA OCR candidate, never proof of server acceptance; no Google grid support."""
            from .captcha import solve_encoded_captcha

            return solve_encoded_captcha(
                image_base64,
                backend=backend,
                expected_length=expected_length,
                alphabet=alphabet,
                challenge_type=challenge_type,
            )

    if recaptcha_config:

        @mcp.tool()
        def recaptcha_solve(target_url: str, allow_network: bool = False, max_seconds: float = 90):
            """Interact with an exact existing Google reCAPTCHA v2 tab using local models; no form submission or server-acceptance guarantee."""
            from .recaptcha import solve_google_recaptcha

            return solve_google_recaptcha(
                target_url, allow_network=allow_network, max_seconds=max_seconds, **recaptcha_config
            )

    if oracle.local_only:
        mcp.remove_tool("web_search")
    return mcp
