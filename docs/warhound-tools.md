# Local WarHound tool integration

Oracle integrates the public WarHoundAI projects through their published HTTP APIs. Each service remains independently deployed and usable. Audited revisions and license hashes are in [tools24](../research/experiments/tools-24/repositories.json); all four repositories declare Apache-2.0, with separate third-party notices. Oracle does not vendor them or assume compatible in-process Python dependencies.

| Service | Oracle use | Local setting |
| --- | --- | --- |
| [HoundSearch](https://github.com/warhoundai/HoundSearch) | Indexed-source lookup before SearXNG | `HOUNDSEARCH_ENDPOINT`, `HOUNDSEARCH_API_KEY_FILE` |
| [HoundCrawler](https://github.com/warhoundai/HoundCrawler) | Single-page acquisition, durable bounded crawl and result import | `HOUNDCRAWLER_ENDPOINT`, `HOUNDCRAWLER_API_KEY_FILE` |
| [HoundScraper](https://github.com/warhoundai/HoundScraper) | Alternative single-page acquisition service | `HOUNDSCRAPER_ENDPOINT`, `HOUNDSCRAPER_API_KEY_FILE` |
| [HoundBrowser](https://github.com/warhoundai/HoundBrowser) | Scoped read-only navigation/observation | `HOUNDBROWSER_ENDPOINT`, `HOUNDBROWSER_API_KEY_FILE` |

Run services using their own repository's setup instructions and generated instance credentials. These are keys for your local services, not vendor keys. Save each key in a separate owner-only file (`chmod 600`). Equivalent `_API_KEY` environment variables are supported; never commit them. Use loopback HTTP or HTTPS. The default ports of some products overlap: give HoundBrowser a free port such as8084 and keep SearXNG on8080. Native commands below reach services on the same machine; a container's127.0.0.1 points inside that container, so use a configured HTTPS gateway or run Oracle natively for these integrations.

```bash
export HOUNDCRAWLER_ENDPOINT=http://127.0.0.1:8081
export HOUNDCRAWLER_API_KEY_FILE=/absolute/private/crawler.key
export HOUNDSEARCH_ENDPOINT=http://127.0.0.1:8082
export HOUNDSEARCH_API_KEY_FILE=/absolute/private/search.key
# Use your own indexed HoundSearch collection with PRIMARY_PROVIDER unset.
houndoracle --web search 'Python release history' --allow-network
houndoracle crawl https://www.python.org/downloads/ --allow-network
houndoracle crawl-site https://www.python.org/downloads/ --max-pages 3 --allow-network
# Read the returned job id. Repeat with next_cursor to import further result pages.
houndoracle crawl-results YOUR_JOB_ID --cursor 0
```

HoundSearch requests use fast local keyword mode, no generated answer, no content fetching, and a downstream cap of one request. This is consumed by local lookup; a configured primary upstream needs another reservation and fails closed instead of silently spending. If the local index has no matches or fails, SearXNG can discover public sources within the remaining query budget. Leave HoundSearch's primary provider unset for this local-index integration.

With HoundCrawler configured, SearXNG uses it to fetch a discovered page. If only HoundScraper is configured, that service is used. `HOUNDORACLE_RENDER=auto` enables the source service's own configured browser worker for JavaScript pages; the default is`never`. Configure the correct worker profile according to that repository. Its `/v1/browser/scrape` protocol differs from standalone HoundBrowser sessions; do not interchange their endpoints.

Bounded durable jobs require HoundCrawler: HoundScraper at the reviewed revision lacks the same scope/concurrency fields, so Oracle refuses its multi-page job path. Durable crawls default to20 pages maximum, depth2, concurrency1,60seconds,5MB aggregate download budget, no external links/subdomains, no sitemap expansion. The server enforces its own per-origin pacing and robots policy. Job creation uses an idempotency key and is never automatically retried after an uncertain response. `crawl-results` imports at most20 records per page, preserves continuation, rejects private literal URLs/challenge pages, and marks all imported evidence unreviewed/non-primary. It can be called while a job is running; an empty current page does not prove completion—inspect the job on its service. Local verification still has its documented corpus-size limits.

For standalone HoundBrowser, explicitly approve the target origin in its operator configuration:

```bash
export HOUNDBROWSER_ENDPOINT=http://127.0.0.1:8084
export HOUNDBROWSER_API_KEY_FILE=/absolute/private/browser.key
houndoracle browser-observe https://example.com/ --allow-network
```

Oracle checks robots first, creates a GET/HEAD-only origin scope, navigates once, polls under a budget, and attempts session cleanup even when the outcome is unknown. It does not replay uncertain commands, execute page-supplied scripts or automatically expand the origin scope. Observation is not imported as verified evidence. HoundBrowser's exact-origin policy can block third-party resources; it is not a universal Google browser replacement.

## Google and CAPTCHA status

SearXNG now sends explicit`language=en-US`; the supplied settings use the same default. Google access-denied and CAPTCHA reports are surfaced through sanitized provider status. Blocked engines cool down (CAPTCHA10minutes; other failures3minutes), while working engines remain usable. Counters/cooldowns are per process and not a distributed scheduler. HTTP200 alone is not a successful search: the direct Google check returned a JavaScript gate without the expected release URL. The previous EC2 test returned Google403 and DuckDuckGo CAPTCHA while Bing worked. A language fix is not proof those restrictions are resolved.

Acquired challenge pages are quarantined with `CAPTCHA_REQUIRES_INTERACTION` and never used as factual evidence. This is challenge detection and fallback, **not an automatic CAPTCHA solver**. A separate [experimental Google reCAPTCHA v2 helper](google-recaptcha.md) now integrates local image models; reliable live completion has not been established. Adding an unqualified solver would not justify claims of reliable Google retrieval. Use available engines/primary-source crawling and treat unresolved challenges as retrieval failures. No accuracy improvement or99% qualification is claimed.

A separate [experimental local text-CAPTCHA recognizer](captcha.md) is now available through CLI/API/MCP. It returns candidates for cropped text images; it does not solve Google interactive challenges or automatically submit forms. The separate v2 helper has its own licensing, setup and acceptance limits.
