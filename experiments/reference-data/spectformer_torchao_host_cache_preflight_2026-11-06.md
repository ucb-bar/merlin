# Exact SpectFormer TorchAO host cache preflight, 2026-11-06

This was an **empty-cache refusal**, not a host inference or numerical gate.
The command used the materialized capture
`spectformer_torchao_w8a8_20261106` (MLIR SHA-256
`9ae3f772ed2d192899cea502e5dec0098d5abf95f56f1707c7ba3cdfcf5481ab`)
and the `kodiak_prepack_quant_w8a8_v1` host package.

The task-local probe exported a live LLVM 23 install as
`MERLIN_MLIR_INSTALL`, with its `clang-23` and `mlir-translate` named by
`MERLIN_CLANG` and `MERLIN_MLIR_TRANSLATE`; it selected the managed
Model2MLIR compiler Python. The root worktree's ignored `.env` instead names
removed LLVM paths, so explicit overrides are necessary here.
The selected `clang-23` SHA-256 is
`40b44020de84a10b8b613d37b96157e88f8433a000bdeb2ad9bdd79af6785125`,
and `mlir-translate` SHA-256 is
`40ae9baf01a5966d596b07e16f4094a6d5fc6f32c908de8a46ecc3d567fe1b4d`.

`merlin-compile --workload spectformer_torchao_w8a8 --target rvv --dtype
int8_w8a8 --capture-bundle <exact capture> --package
out/artifacts/targets/rvv/kodiak_prepack_quant_w8a8_v1 --run host --verify
--host-cache-dir <empty task cache> --json` exited 1 in 9.35 seconds with
`status=error`, `HostCacheMiss`, `miss_count=1361`, `loaded_unique=0`, and
no compiler invocation. Compiler identity SHA-256 was
`6ed34eebca66820cf9dc7eed42877c8175354c7612677118dceef28faca6b4cd`.
The exact JSON is retained in the isolated `source-w8a8-cache-miss` managed
task under `.tmp/spect-cache-miss.json`.

This proves the strict path refuses stale or absent objects before execution.
It does not change the prior complete SpectFormer host gate failure. Fresh
receipted object population and a complete framework comparison are required
before exact-source OPU or VME model acceptance.
