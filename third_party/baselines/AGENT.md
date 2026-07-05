# AGENT.md — third_party/baselines

## Purpose

External **baseline compilers/runtimes** that merlin is compared against. Each ingests the *same*
models we support and runs them end-to-end on the *same* SpacemiT K1 board with **RVV**, on its own
stack — so we can position merlin honestly against independent toolchains (not just a kernel swap
inside merlin's own runtime).

## What belongs here

Pinned **submodules** of external frameworks used purely for cross-framework K1 comparison:

| Submodule    | Upstream                                   | Role in the comparison                               |
|--------------|--------------------------------------------|------------------------------------------------------|
| `buddy-mlir` | github.com/buddy-compiler/buddy-mlir       | MLIR compiler — ingests our `model.mlir` → RVV       |
| `tvm`        | github.com/apache/tvm                      | Relax/Relay import + Ansor/MetaSchedule autotune→RVV |
| `executorch` | github.com/pytorch/executorch              | torch.export → `.pte` + XNNPACK delegate (RVV)       |
| `exo`        | github.com/exo-lang/exo                    | Kernel DSL + scheduler/autotuner (RVV kernels)       |
| `llama.cpp`  | github.com/ggml-org/llama.cpp              | ggml runtime, RVV kernels (LLM subset + bitvla)      |

## Why submodules (carve-out from `third_party/AGENT.md`)

The parent `third_party/AGENT.md` reserves `third_party/` for hard build deps and says external
analysis repos are "integrations" reached via env-var. `baselines/` is a **deliberate exception**:
for a reproducible cross-framework benchmark, the *exact framework commit* is part of the
measurement, so these are pinned submodules rather than floating env-var checkouts. Added shallow
(`--depth 1`); the superproject gitlink records the pinned SHA.

## Interfaces

Driven by the shared harness `merlin/python/merlin/baselines/` (bundle resolution, K1 exec + board
lock, RVV-coverage audit, whole-model + per-region profiling, result contract) and surfaced through
`merlin-compare`. Per-framework runner logic lives in `merlin/python/merlin/baselines/<framework>.py`.

## Invariants

- **Push maximal RVV**; where an op/region falls back to scalar, that is **labeled explicitly**
  (per-op / per-region) via the objdump-based RVV-coverage audit — never hidden or averaged away.
- **`not_run_is_not_pass`**: a model that doesn't compile/run on a framework is an explicit gap, not
  an omission.
- Build trees go in `build/<framework>/` (gitignored); results in `artifacts/compare/` +
  `artifacts/measurements/k1_spacemit/<model>/`. Nothing generated is committed under this tree.
- Nested submodules (e.g. ExecuTorch's XNNPACK, TVM's 3rdparty) are init'd per-arm at build time
  (`git submodule update --init --recursive <path>`), not eagerly.

## Testing expectations

Board-free unit tests (march enforcement, RVV classifier, bundle/contract) must stay green without a
live board. On-board runs are serialized (single physical K1) and fail-closed when `MERLIN_K1_HOST`
is unset/unreachable.

## Notes for future agents

- Toolchain: SpacemiT clang (`$MERLIN_K1_TOOLCHAIN`), `-march=rv64gcv -mabi=lp64d`, VLEN=256, glibc
  Linux (not bare-metal). See `merlin/python/merlin/rvvgen/k1.py`.
- IREE is intentionally **deferred** to a later pass (not yet a submodule here).
