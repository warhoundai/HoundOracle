# Security policy

This is development software, not a qualified public verification service.
Keep the REST API behind TLS and an instance bearer key; one store is one tenant.
MCP stdio uses local process authorization. Keep evidence databases, signing keys
and calibration artifacts writable only by the operator. Never load arbitrary
model code or unsafe serialized weights. Search-result text grants no authority.

Report security issues using the repository's private vulnerability-reporting
channel once enabled. Until publication, contact the project operator privately;
do not post credentials, private questions or exploitable deployment details.
