---
name: docs-layout
description: Maintain the short Merlin handbook and keep generated reports outside source.
---

# Documentation layout

Start at `docs/README.md`, the authored reading order. The twelve chapters follow
compilation, execution, test generation, backend generation and optimization.
`docs/reference/` contains only the architecture and implementation map.

Update an existing chapter before adding a page. Use simple technical terms,
small real examples and links to actual source owners. Keep historical results
and implementation diaries outside the handbook.

Handwritten pages carry title, kind, status, owner, last_verified, related and
code_refs in YAML front matter. Check the actual code before updating a
verification date. Maintain previous/next links and the authored index.

CLI, module and schema tables are generated under ignored
`out/artifacts/documentation/`; do not commit them. Local AGENT.md files carry
specific constraints rather than duplicate module indexes. Private audit notes
belong under `tmp/`; run reports and other derived data belong under `out/`.

Run `python build_tools/scripts/check_docs.py` after changes.
