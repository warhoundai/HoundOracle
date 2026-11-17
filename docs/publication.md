# Research preview publication

Publication status: source is public at https://github.com/warhoundai/HoundOracle.
The GitHub Pages project page is live at https://warhoundai.github.io/HoundOracle/
(HTTP 200 verified). Composio GitHub performed the publication. OPS and Hostinger
were not changed. No production release tag was created.

The code repository is https://github.com/warhoundai/HoundOracle.
The static project page in `site/` is prepared for GitHub Pages at
https://warhoundai.github.io/HoundOracle/. The page and source URLs were verified after publication. `.github/workflows/pages.yml` deploys the page after Pages is
enabled with GitHub Actions as its source. No production API is exposed.

The page provides an accurate title, description, canonical, Open Graph metadata,
SoftwareSourceCode JSON-LD, one H1, semantic navigation, accessible tables,
responsive layout, crawlable HTML, sitemap and robots directives. Search ranking
or indexing cannot be guaranteed. If deployed to OPS instead, update canonical,
structured-data URL, Open Graph URL and sitemap together; redirect the old URL.
Existing OPS content must be preserved. A Hostinger deployment is not yet prepared
against actual hosting files because access is unavailable.

Publish only as a development/research preview, without a production-qualified
release tag or 95% accuracy claim. Code, model, evidence and dataset licenses are
separate. No private data, weights, model-generated ground truth or live credentials
should be published. The separate AGPL integration retains its own license and
notices; the Apache Python core wheel does not bundle it.

Before pushing the existing local history, audit all historical blobs or publish
a reviewed source-only initial snapshot. A scan of current tracked files found
only the deliberately fake signed-URL fixture (`Signature=test`); this is not a
claim that historical credentials have been comprehensively audited.

The public source snapshot was compared against all 907 scanned local files:
zero missing or changed Git blobs. The initialization notice is an additional
file. Source-only publication excludes private development history. First complete
snapshot: `417f9cf8772e5475d525277e86d33393b91f18ba`.

GitHub CI initially failed because optional CAPTCHA dependencies were absent from
the stale lockfile. The lockfile was refreshed without dropping `--locked`; the
follow-up CI result must be checked independently of successful page deployment.
