# Integrations

Integrations are **adapters**, never vendored repositories. Adapters live **in-package** under
`merlin/python/merlin/` (e.g. `kernels/` already ingests several external kernel sources this way) —
there is no standalone `merlin/integrations/` tree. See `docs/design/integrations.md` for the
rationale and how to add one.

| Dependency type   | Where                          | When                                          |
| ----------------- | ------------------------------ | --------------------------------------------- |
| Adapter only      | in-package (`merlin/python/merlin/…`) | parse/index/emit-to/call the external tool |
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

Each adapter normalizes its source into `merlin/schemas/` artifacts (e.g. `kernel_record`,
`abstraction_candidate`, `policy_rule`) so all sources are comparable.

Note: xDSL is consumed as an optional library (adapters call its *tooling*); merlin's own prototype
dialects live in `merlin/python/merlin/xdsl_dialects/`.
