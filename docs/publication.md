# Research preview publication

Publication status: public research source snapshot prepared for GitHub. GitHub authentication is
`warhoundai`, which also owns the four released Hound repositories. No organization
or legal ownership relationship is inferred. The user created the public HoundOracle repository after repository creation
was denied to the integration. Publication is GitHub-only; OPS is unchanged.

The intended code repository is https://github.com/warhoundai/HoundOracle.
The static project page in `site/` is prepared for GitHub Pages at
https://warhoundai.github.io/HoundOracle/. Repository and page deployment verification is recorded in the publication
handoff; the page is not live until its deployment succeeds. `.github/workflows/pages.yml` deploys the page after Pages is
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

The initial public commit is a source-only snapshot of local commit
`62c4b83`. Private development history is deliberately not pushed. Research
metadata, licenses and experiment reports are retained; models and private
evidence remain outside this repository.
