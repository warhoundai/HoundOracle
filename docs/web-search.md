# Self-hosted Oracle with internet access

Oracle runs on the user's machine. Its SQLite index, evidence snapshots, verification and inference remain local. A self-hosted SearXNG service provides web discovery through Google, Bing and DuckDuckGo; no paid search API, Gemini account or central Oracle service is required. This is the intended internet-enabled profile. The separate offline profile is available for disconnected use.

Google's index and ranking still run at Google. SearXNG sends the query to upstream engines; it is not a downloadable Google engine. Those engines see search queries and the machine's public IP and can rate-limit, present CAPTCHAs or stop serving an adapter. Respect their applicable terms and access restrictions. Do not bypass challenges or promise uninterrupted Google access. Check `unresponsive_engines` when investigating empty results; an outage is not proof of falsehood.

## Run the complete stack

```bash
export HOUNDORACLE_API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
docker compose -f deployment/selfhost/compose.web.yaml up --build -d
curl http://127.0.0.1:8083/v1/search \
  -H "Authorization: Bearer $HOUNDORACLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query":"Python 3.12.0 release date site:python.org","allow_network":true,"max_requests":4}'
```

The token protects your own local API; it is not a vendor credential. The search service is reachable only inside the Compose network, and Oracle's published API binds to loopback. The web network intentionally permits internet access. Stop with the same Compose file and `down`; the evidence volume persists. Avoid running the offline and web stacks simultaneously on the same port/database.

The pinned search image is linux/amd64; users of another architecture must select and verify an appropriate upstream image digest. SearXNG is a separate AGPL-3.0-or-later service, not Apache-licensed HoundOracle code. Its upstream source and licensing are at https://github.com/searxng/searxng. Preserve upstream notices and meet its license obligations when modifying or distributing it. Search results and downloaded pages keep their own rights; crawling does not grant redistribution permission.

## Native Python and MCP

With your own local SearXNG JSON endpoint (enable `search.formats: [html, json]`):

```bash
houndoracle --web --searxng-url http://127.0.0.1:8080 search \
  'Python 3.12.0 release date site:python.org' --allow-network
houndoracle --web --searxng-url http://127.0.0.1:8080 mcp
houndoracle crawl https://www.python.org/downloads/release/python-3120/ --allow-network
```

```python
from houndoracle import Oracle, Store
from houndoracle.web import SearXNGAdapter

oracle = Oracle(Store('oracle.sqlite'), local_only=False,
                provider=SearXNGAdapter('http://127.0.0.1:8080'))
evidence = oracle.web_search('Python 3.12.0 release date site:python.org',
                             allow_network=True, max_requests=4)
```

Only explicitly permitted requests leave the machine. Private questions can use local search without `allow_network`; vendor environment keys do not activate hosted model inference. Fresh local evidence is searched first; `--fresh` bypasses that evidence shortcut. Similarity never authorizes reuse of a factual verdict.

## Acquisition and remaining limits

A search returns at most three deduplicated snippets and attempts one full public source page within the request/time budget. Two additional HTTP requests check robots.txt and obtain the page. Each failed attempt consumes its reservation. Missing robots.txt (404/410) permits fetching; robots failures and disallow rules stop it. Sites with explicit crawl-delay/request-rate policies require a scheduled crawler and are conservatively skipped. Cross-origin redirects are rejected. Network-level DNS checks block private/link-local source targets, including cloud metadata. The private-endpoint exception is restricted to the operator-configured SearXNG origin and is never passed to source fetching.

This bounded seed fetcher does not execute JavaScript, recursively traverse links, parse PDFs or replace the independent HoundCrawl/HoundScrapper products. Source text and search snippets are distinguished in persisted provenance; all remain unreviewed/non-primary. Exact duplicate content is collapsed by the existing evidence store. SearXNG performs multiple upstream requests internally: Oracle's HTTP request count is not a count of individual Google/Bing requests. The configuration restricts engines to three and applies upstream timeouts. No paid API charge does not mean zero bandwidth/compute cost.

Retrieved text cannot change tool permissions or become a model instruction. Default verification still abstains without qualifying calibration and supports a narrow deterministic claim grammar. Adding web search has not established improved accuracy; 99–100% remains an unmet target requiring independent retrieval/evidence/verdict evaluation.

Live EC2 smoke results: [web23 report](../research/experiments/web-23/README.md). Google denied access and DuckDuckGo returned CAPTCHA from that IP; Bing returned results. This setup cannot promise universal Google availability.

For HoundSearch, HoundCrawler, HoundScraper and HoundBrowser adapters, see [WarHound integrations](warhound-tools.md). Google requests now pin en-US; failed engines cool down and expose sanitized status. This does not establish that Google access restrictions are fixed.

A separate [experimental Google reCAPTCHA v2 helper](google-recaptcha.md) can interact with an existing local browser; solving there does not clear SearXNG's independent HTTP session or establish reliable Google access.
