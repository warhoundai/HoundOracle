import argparse
import json
import os
from pathlib import Path

from .calibration import load_calibration
from .engine import Oracle
from .models import DocumentInput
from .store import Store


def main():
    parser = argparse.ArgumentParser(description="HoundOracle development preview")
    parser.add_argument("--database", default="oracle.sqlite")
    parser.add_argument("--calibration")
    parser.add_argument(
        "--semantic-model", type=Path, help="Local general-claim NLI weights; no downloads"
    )
    parser.add_argument("--semantic-weights-sha256", help="Required checkpoint integrity pin")
    parser.add_argument("--semantic-adapter", type=Path)
    parser.add_argument("--semantic-adapter-sha256")
    parser.add_argument("--recaptcha-helper-dir", type=Path)
    parser.add_argument("--recaptcha-model-dir", type=Path)
    parser.add_argument("--recaptcha-cdp-port", type=int, default=9222)
    parser.add_argument(
        "--captcha", action="store_true", help="Enable experimental local text-CAPTCHA API/MCP tool"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Use your self-hosted SearXNG for permitted live requests",
    )
    parser.add_argument(
        "--searxng-url", default=os.environ.get("HOUNDORACLE_SEARXNG_URL", "http://127.0.0.1:8080")
    )
    sub = parser.add_subparsers(dest="command", required=True)
    records = sub.add_parser("release-record", help="Experimental exact Node.js release-index tool")
    records.add_argument("question")
    records.add_argument("--allow-network", action="store_true")
    records.add_argument("--use-environment-proxy", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("question")
    verify.add_argument("--max-cost-usd", type=float, default=0)
    verify.add_argument("--allow-network", action="store_true")
    verify.add_argument("--min-confidence", type=float, default=0.8)
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--allow-network", action="store_true")
    search.add_argument("--fresh", action="store_true")
    search.add_argument("--max-cost-usd", type=float, default=0)
    ingest = sub.add_parser(
        "ingest", help="Local curator action: imports trust flags from supplied JSON"
    )
    ingest.add_argument("document", type=Path)
    export = sub.add_parser("export")
    export.add_argument("destination", type=Path)
    imp = sub.add_parser("import")
    imp.add_argument("snapshot", type=Path)
    explain = sub.add_parser("explain")
    explain.add_argument("result_id")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8083)
    sub.add_parser("mcp")
    index = sub.add_parser(
        "index",
        help="Build local evidence search from JSONL snapshots; never calls a hosted service",
    )
    index.add_argument("documents", type=Path)
    crawl = sub.add_parser(
        "crawl", help="Fetch a public seed URL into local evidence, respecting robots.txt"
    )
    crawl.add_argument("url")
    crawl.add_argument("--allow-network", action="store_true")
    browser = sub.add_parser(
        "browser-observe", help="Read-only observation through your local HoundBrowser"
    )
    browser.add_argument("url")
    browser.add_argument("--allow-network", action="store_true")
    site = sub.add_parser(
        "crawl-site", help="Submit a bounded durable job to local HoundCrawler/HoundScraper"
    )
    site.add_argument("url")
    site.add_argument("--max-pages", type=int, default=20)
    site.add_argument("--allow-network", action="store_true")
    results = sub.add_parser(
        "crawl-results", help="Import one page of local crawl results as unreviewed evidence"
    )
    results.add_argument("job_id")
    results.add_argument("--cursor", type=int, default=0)
    captcha = sub.add_parser(
        "captcha-solve", help="Experimental local text-image CAPTCHA recognition"
    )
    captcha.add_argument("image", type=Path)
    captcha.add_argument("--backend", choices=["ddddocr", "tesseract"], default="tesseract")
    captcha.add_argument("--length", type=int)
    captcha.add_argument(
        "--alphabet", default="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    captcha.add_argument("--max-seconds", type=float, default=20)
    recaptcha = sub.add_parser(
        "recaptcha-solve",
        help="Experimental local Google reCAPTCHA v2 helper on an existing dedicated Chrome tab",
    )
    recaptcha.add_argument("target_url")
    recaptcha.add_argument("--helper-dir", type=Path, required=True)
    recaptcha.add_argument("--model-dir", type=Path, required=True)
    recaptcha.add_argument("--cdp-port", type=int, default=9222)
    recaptcha.add_argument("--max-seconds", type=float, default=90)
    recaptcha.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    if args.command == "release-record":
        from .tools.releases import ReleaseRecords

        options = {}
        if args.use_environment_proxy:
            from .tools.official_transport import fetch_official

            options["transport"] = fetch_official
        tool = ReleaseRecords(args.database + ".releases", **options)
        try:
            print(
                json.dumps(tool.verify(args.question, allow_network=args.allow_network), indent=2)
            )
        finally:
            tool.close()
        return
    recaptcha_config = None
    if args.recaptcha_helper_dir or args.recaptcha_model_dir:
        if not (args.recaptcha_helper_dir and args.recaptcha_model_dir):
            parser.error("Both --recaptcha-helper-dir and --recaptcha-model-dir are required")
        recaptcha_config = {
            "helper_directory": args.recaptcha_helper_dir,
            "model_directory": args.recaptcha_model_dir,
            "port": args.recaptcha_cdp_port,
        }
    store = Store(args.database)
    from .search import FederatedSearch
    from .warhound import HoundIndexAdapter, service_key, source_from_environment
    from .web import SearXNGAdapter

    provider = None
    if args.web:
        web = SearXNGAdapter(
            args.searxng_url,
            source_acquirer=source_from_environment(),
            fetch_sources=not bool(args.semantic_model),
        )
        if os.environ.get("HOUNDSEARCH_ENDPOINT"):
            index = HoundIndexAdapter(
                os.environ["HOUNDSEARCH_ENDPOINT"], service_key("HOUNDSEARCH")
            )
            provider = FederatedSearch([index, web])
        else:
            provider = web

    general = None
    if args.semantic_model is not None:
        if not args.semantic_weights_sha256:
            parser.error("--semantic-model requires --semantic-weights-sha256")
        from .general import GeneralVerifier
        from .local_semantics import LocalPairEvaluator

        if bool(args.semantic_adapter) != bool(args.semantic_adapter_sha256):
            parser.error("Adapter weights and integrity pin must be supplied together")
        evaluator = LocalPairEvaluator(
            args.semantic_model,
            expected_weights_sha256=args.semantic_weights_sha256,
            adapter_path=args.semantic_adapter,
            expected_adapter_sha256=args.semantic_adapter_sha256,
        )
        acquirer = source_from_environment()
        general = GeneralVerifier(
            evaluator,
            **(
                {"source_acquirer": acquirer, "acquirer_version": type(acquirer).__name__}
                if acquirer
                else {}
            ),
        )
    elif args.semantic_weights_sha256 or args.semantic_adapter or args.semantic_adapter_sha256:
        parser.error("--semantic-weights-sha256 requires --semantic-model")
    oracle = Oracle(
        store,
        calibration=load_calibration(args.calibration) if args.calibration else None,
        provider=provider,
        local_only=not args.web,
        general_verifier=general,
    )
    try:
        if args.command == "recaptcha-solve":
            from .recaptcha import solve_google_recaptcha

            print(
                json.dumps(
                    solve_google_recaptcha(
                        args.target_url,
                        helper_directory=args.helper_dir,
                        model_directory=args.model_dir,
                        port=args.cdp_port,
                        max_seconds=args.max_seconds,
                        allow_network=args.allow_network,
                    )
                )
            )
        elif args.command == "captcha-solve":
            from .captcha import solve_text_captcha

            with args.image.open("rb") as stream:
                image = stream.read(1_000_001)
            print(
                json.dumps(
                    solve_text_captcha(
                        image,
                        backend=args.backend,
                        expected_length=args.length,
                        alphabet=args.alphabet,
                        max_seconds=args.max_seconds,
                    )
                )
            )
        elif args.command == "verify":
            print(
                oracle.verify(
                    args.question,
                    allow_network=args.allow_network,
                    max_cost_usd=args.max_cost_usd,
                    min_confidence=args.min_confidence,
                ).model_dump_json(indent=2)
            )
        elif args.command == "search":
            print(
                json.dumps(
                    oracle.web_search(
                        args.query,
                        allow_network=True,
                        max_requests=4,
                        prefer_local=not args.fresh,
                        max_cost_usd=args.max_cost_usd,
                    )
                    if args.allow_network
                    else oracle.search(args.query),
                    indent=2,
                )
            )
        elif args.command in {"crawl-site", "crawl-results"}:
            from .budget import Budget
            from .warhound import source_from_environment

            crawler = source_from_environment()
            if crawler is None:
                parser.error(
                    "Configure HOUNDCRAWLER_ENDPOINT or HOUNDSCRAPER_ENDPOINT and a local service key"
                )
            budget = Budget(1, 20, 0)
            if args.command == "crawl-site":
                if not args.allow_network:
                    parser.error("crawl-site requires --allow-network")
                print(json.dumps(crawler.submit_crawl(args.url, budget, max_pages=args.max_pages)))
            else:
                import time
                from urllib.parse import urlsplit

                page = crawler.crawl_results(args.job_id, budget, cursor=args.cursor)
                ids = []
                for hit in page["results"]:
                    now = time.time()
                    ids.append(
                        store.ingest(
                            DocumentInput(
                                url=hit["url"],
                                text=hit["text"],
                                origin=urlsplit(hit["url"]).hostname,
                                observed_at=now,
                                expires_at=now + 3600,
                                reviewed=False,
                                primary=False,
                                retrieved_by=crawler.name,
                                retrieval_kind="source_document",
                            )
                        )
                    )
                print(
                    json.dumps(
                        {"document_ids": ids, "next_cursor": page["next_cursor"], "verified": False}
                    )
                )
        elif args.command == "browser-observe":
            if not args.allow_network:
                parser.error("browser-observe requires --allow-network")
            from .budget import Budget
            from .warhound import HoundBrowserAdapter

            browser = HoundBrowserAdapter(
                os.environ.get("HOUNDBROWSER_ENDPOINT", "http://127.0.0.1:8084"),
                service_key("HOUNDBROWSER"),
            )
            print(json.dumps(browser.observe(args.url, Budget(20, 30, 0))))
        elif args.command == "crawl":
            if not args.allow_network:
                parser.error(
                    "crawl requires --allow-network; this sends requests to the public source"
                )
            import time
            from urllib.parse import urlsplit

            from .budget import Budget
            from .warhound import source_from_environment
            from .web import acquire_source

            acquire = source_from_environment() or acquire_source
            hit = acquire(args.url, Budget(2, 20, 0))
            if not hit:
                parser.error("source contains no usable text")
            now = time.time()
            identifier = store.ingest(
                DocumentInput(
                    url=hit["url"],
                    text=hit["text"],
                    origin=urlsplit(hit["url"]).hostname,
                    observed_at=now,
                    expires_at=now + 3600,
                    reviewed=False,
                    primary=False,
                    retrieved_by=hit.get("retrieval_provider", "local-crawler"),
                    retrieval_kind="source_document",
                )
            )
            print(json.dumps({"document_id": identifier, "verified": False}))
        elif args.command == "index":
            from .local_index import import_documents

            print(json.dumps(import_documents(store, args.documents)))
        elif args.command == "ingest":
            print(store.ingest(DocumentInput.model_validate_json(args.document.read_text())))
        elif args.command == "export":
            args.destination.write_text(json.dumps(store.export(), indent=2))
        elif args.command == "import":
            print(json.dumps(store.import_snapshot(json.loads(args.snapshot.read_text()))))
        elif args.command == "explain":
            print(json.dumps(oracle.explain(args.result_id), indent=2))
        elif args.command == "serve":
            import uvicorn

            from .api import create_app

            uvicorn.run(
                create_app(
                    oracle,
                    api_key=os.environ.get("HOUNDORACLE_API_KEY"),
                    enable_captcha=args.captcha,
                    recaptcha_config=recaptcha_config,
                ),
                host=args.host,
                port=args.port,
            )
        elif args.command == "mcp":
            from .mcp import server

            server(oracle, enable_captcha=args.captcha, recaptcha_config=recaptcha_config).run(
                transport="stdio"
            )
    finally:
        store.close()
