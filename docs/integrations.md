# Integrations

Integrations are **adapters**, never vendored repositories.

| Dependency type   | Where                          | When                                          |
| ----------------- | ------------------------------ | --------------------------------------------- |
| Adapter only      | `merlin/integrations/<name>/`  | parse/index/emit-to/call the external tool    |
| External checkout | outside the repo, by path/env  | inspect or run the repo; merlin doesn't own it|
| Pinned dependency | `third_party/<name>/`          | merlin cannot build/test without it           |

Decision rule: *if merlin can run without it, it is an integration; if merlin cannot build without
it, it is third_party.*

External repos are passed by env var:

```bash
export MERLIN_XNNPACK_REPO=/path/to/XNNPACK
export MERLIN_AUTOCOMP_REPO=/path/to/autocomp
export MERLIN_EXO_REPO=/path/to/exo
export MERLIN_TRITON_REPO=/path/to/triton
```

Each adapter (`README.md`, `AGENT.md`, `manifest.yaml`) normalizes its source into
`merlin/schemas/` artifacts so all sources are comparable.

Note: `merlin/integrations/xdsl/` is an adapter to xDSL *tooling*; merlin's own prototype dialects
live separately in `merlin/python/merlin/xdsl_dialects/`.
