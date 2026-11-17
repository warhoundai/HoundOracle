# WarHound tool adapters and challenge-aware retrieval

User request: investigate public WarHound repositories, integrate tools, fix Google and add automatic CAPTCHA solving. Repository identities, revisions and licenses were checked through GitHub API, README, original models, engine code and LICENSE. Pinned inspection metadata: repositories.json.

Implemented HTTP integrations preserve independent operation and avoid cross-project Python dependency coupling: HoundSearch index lookup before web discovery, HoundCrawler/HoundScraper page acquisition, bounded HoundCrawler durable jobs with cursor import, and scoped HoundBrowser observation with pre-reserved cleanup. No hosted inference is invoked. Text/metadata returned by these tools never auto-qualify as factual evidence.

Original request-model validation caught incompatible HoundScraper job fields; the restricted multi-page path now requires HoundCrawler. Actual payloads pass original HoundCrawler scrape/job, HoundScraper scrape and HoundSearch search model classes. See contracts.json. A real loopback HTTP fixture tests HoundBrowser 201 creation, 202 command acceptance, terminal polling and 204 cleanup. Source robots/public-network and uncertain-command replay regressions remain enforced. Live running PostgreSQL/Chromium tool stacks were not provisioned; schema and fixture checks are not a complete cross-service deployment benchmark.

Google investigation: the inspected upstream SearXNG Google engine uses a WML XML path and notes that the regular web endpoint needs JavaScript. Requests now specify en-US instead of relying on unspecified locale. A single direct regular-web request with explicit hl=en returned HTTP200 and a JavaScript enable gate; it contained no expected Python release URL. This does not demonstrate a successful Google result or fix its earlier EC2 access-denied response. Failed engines now have bounded process-local cooldowns, sanitized failure codes and working-engine fallback.

CAPTCHA handling: recognizable challenge pages are quarantined rather than ingested as claims. Browser observation reports CAPTCHA_REQUIRES_INTERACTION; unknown outcomes are not replayed. **No automatic Google CAPTCHA solver has been implemented or validated.** Do not describe challenge detection, cooldown or browser observation as solving a CAPTCHA. This user requirement and reliable Google access remain open. No accuracy improvement or99% verification qualification is established.

No paid cloud worker or public deployment was created for this change. Public Google lookup and GitHub reads were the only live external investigations; original tool stack requests in contract validation were mocked. Existing authorized-budget ledger unchanged. Prior temporary EC2 resources remain terminated. See docs/warhound-tools.md for setup and operational limits.

Verification:177 tests pass on Python3.11/3.12; lint, diff checks and wheel build pass. No new accuracy benchmark was consumed.
