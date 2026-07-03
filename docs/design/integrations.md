# Design note: external-tool integration adapters

**Status: not built. Implement an adapter in-package (`merlin/python/merlin/…`) when a concrete
need arises** — do not keep an empty adapter skeleton.

The original design reserved `merlin/integrations/<tool>/` (XNNPACK, Autocomp, Exo, Triton, xDSL,
IREE, CUDA-Tile, Hexagon-MLIR, OpenBLAS) for lightweight **adapters** that parse/index/normalize an
external project (passed by path/env, never vendored) and emit merlin schema artifacts
(`kernel_record` / `abstraction_candidate` / `policy_rule`, per `merlin/schemas/`). Every dir was
intent-only (`README.md` + `manifest.yaml` + `AGENT.md`, zero `.py`), so it was removed.

**When implementing one:** put the adapter under `merlin/python/merlin/` (e.g. a `kernels/ingest/`
source or a small `integrations` subpackage), gated by an env var (`MERLIN_<TOOL>_REPO`), emitting
the normalized schema artifacts. Kernel-mining already ingests several external kernel sources this
way under `merlin/python/merlin/kernels/`.
