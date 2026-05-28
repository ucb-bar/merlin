# `kernels/` — kernel-embedding framework

Python library that the `./merlin compile` pipeline imports to substitute
MLIR dispatches with pre-built per-backend kernel artifacts. **Not a
tool** — there is no `./merlin kernels` subcommand. The library is invoked
from `tools/compile/qnn.py` and friends.

This directory was extracted from `tools/kernels/` in May 2026 — it didn't
belong under `tools/` (it's infrastructure, not a developer CLI).

## Layout

```
kernels/
├── core/                  backend-agnostic pipeline
│   ├── discover.py        linalg-op discovery → manifest skeleton
│   ├── manifest.py        manifest schema + loader
│   ├── precompile.py      source-language → object dispatch
│   └── spec_gen.py        transform-dialect spec emitter
├── qnn/                   Qualcomm QNN backend
│   ├── build.py emit.py ir.py partition.py route.py gates.py
│   ├── precompile_extras.py
│   ├── recognizers/       pattern matchers (1 file per op family)
│   ├── headers/           C++ headers for runtime integration
│   └── tests/             QNN test suite
└── spike/                 RISC-V Spike simulator runner
    └── runner.py
```

## Adding a new backend

Create `kernels/<backend>/` with at minimum:
- `build.py` — toolchain orchestration that produces object artifacts.
- (optional) `emit.py` — MLIR → backend-IR emitter if the backend needs
  graph-level lowering.
- (optional) `precompile_extras.py` — hook called from `core.precompile`
  when the manifest's `source_lang` matches a value your backend owns.

Register the new `source_lang` in `core/manifest.py:_VALID_SOURCE_LANGS`.

## Status

- **QNN**: v1 path (`qnn.emit`) is active. A v2 emitter attempt was
  archived to `tools/archive/qnn_v2/` after we couldn't get it to working
  parity with v1; see that folder's README.
- **Spike**: small, self-contained, stable.
- **Gemmini / SaturnOPU**: not in `kernels/` today. Their kernel-embedding
  flows live in the relevant target's `models/<target>.yaml` flag set or
  through `tools/compile/qnn.py` (for QNN-routed dispatches).
