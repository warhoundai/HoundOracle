# Rights register

Reviewed 2026-11-15. This is an engineering inventory, not a legal opinion.

| Material | Status / intended handling |
|---|---|
| Oracle code and synthetic fixtures | Original implementation, Apache-2.0; no model or dataset bundled |
| FIRE repository | GitHub reports no license; code read for analysis only, not copied or distributed |
| FIRE paper | Original paper cited; figures/text not redistributed |
| Stockfish / nnue-pytorch | GPL family; no source, weights or implementation fragments incorporated |
| AlphaZero / MuZero / AlphaGeometry papers | Research references; no code or weights incorporated |
| Deep Sets / Set Transformer / calibration research | Research references; implementations independently written |
| AVeriTeC | Official page links CC BY-NC 4.0; excluded from commercial-context training/evaluation/redistribution pending professional review |
| FEVER | Downloaded outside the repository for bounded research; original human labels cross-checked against a pinned gold-evidence mirror. FEVER/Wikipedia attribution-share-alike and mirror GPL declarations remain separate from Apache code. Full question/evidence text is not redistributed. See `../research/experiments/dataset-manifest.json` and `../src/houndoracle/experiments/dataset.py`. |
| FEVEROUS / HoVer | Not downloaded or redistributed; verify licenses before future experiments. |
| Public webpages | Public accessibility does not grant redistribution rights; central publication requires explicit permission and a license allowed by the operator |
| Hugging Face model weights | Pinned `cross-encoder/nli-MiniLM2-L6-H768` downloaded outside the source distribution; model card declares Apache-2.0, base/training-data rights are separate. File hashes and revision are in `../research/model-comparisons/minilm-provenance.json`; no weights redistributed. |
| Local synthetic benchmark | Generated examples, no external dataset; unsuitable for broad accuracy claims |

Runtime core uses Pydantic (MIT). Optional API uses FastAPI (MIT), Starlette (BSD),
Uvicorn (BSD); MCP Python SDK (MIT); S3 uses boto3/botocore (Apache-2.0) and
cryptography (Apache-2.0 OR BSD); semantic experiments use Transformers
(Apache-2.0) and PyTorch (BSD family). These are package-level declarations, not
an exhaustive transitive clearance. `uv.lock` pins the dependency resolution and
`scripts/license_inventory.py` records installed distribution metadata. Unknown,
noncommercial or incompatible declarations require review before release.

Do not bundle the downloaded research PDFs, FIRE source snapshots, or website
HTML. They remain outside the repository in `/workspace/research-sources` for
inspection; `docs/source-manifest.json` records retrieval hashes and URLs. Model
weight provenance, dataset attribution/share-alike requirements, and source text
permissions remain release gates.
