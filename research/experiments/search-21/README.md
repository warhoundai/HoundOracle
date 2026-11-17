# Hosted search research smoke tests

Before the fully local requirement, Exa and Wikipedia adapters were implemented and exercised. Exa returned three Python primary-source pages per smoke request and reported $0.007 per request (two successful requests). A CLI search imported unreviewed evidence; subsequent SQLite discovery avoided another live call. Live lookup was about 2.03 seconds and cached discovery about 1.54 milliseconds in this smoke test. These are individual retrieval observations, not paired accuracy or economics benchmarks.

Provider budgets, credential isolation, duplicate suppression, restricted endpoints and explicit online permission are tested. The supplied key remains outside the repository and bundles. Hosted adapters are research-only: the required self-hosted API/MCP profile does not use them. Search snippets and agreement do not establish truth. See ../local-22/README.md for the current local-only deployment and unmet accuracy target.
