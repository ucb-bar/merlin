---
name: test-layout
description: >-
  Where to put tests in merlin and how to name them. Use whenever you add, move, or organize
  a test, write a new test_*.py, or wonder which folder a test belongs in. One suite at
  merlin/tests/, organized by subsystem bucket.
---

# Test layout (MANDATORY)

All tests live in **`merlin/tests/`** (the only pytest `testpaths`), in **subsystem buckets**:

| Bucket | Put tests for… |
|--------|----------------|
| `kernels/` | kernel mining/ceiling/CCA/policy/features, kernel backend |
| `rvv/` | RVV codegen (rvvgen), RVV/K1 board bringup, model-on-RVV |
| `dse/` | DSE tools (dse / dse_guidance / design_pressure), cost model, search, compare |
| `gemmini/` | Gemmini conformance/cert, RTL checks, OOT runner, bench contract |
| `targetgen/` | TargetGen synthesis + contract validation |
| `ir/` | xDSL dialects, lowering/passes, dispatch, frontends, llvmlower |
| `runtime/` | runtime backends (spike/zephyr/xnnpack/openblas/saturn), engine |
| `infra/` | repo conventions (artifact layout, smoke / CLI smoke) |

## Rules (enforced by `build_tools/scripts/check_structure.py` "test layout")

- Path: `merlin/tests/<bucket>/test_<area>.py`. **Never** put a `test_*.py` at the `merlin/tests/` root,
  and `<bucket>` must be one of the eight above (add a new bucket only by also adding it to
  `TEST_BUCKETS` in `check_structure.py`).
- Place a test in the subsystem folder it exercises.
- Shared inputs: `merlin/tests/fixtures/`, `merlin/tests/data/`.
- Resolve repo paths via `from merlin.common.paths import repo_root, merlin_dir` — **never**
  `Path(__file__).resolve().parents[N]` (keeps tests location-independent).

Run: `.venv/bin/python -m pytest merlin/tests` (or a bucket: `pytest merlin/tests/dse`).
