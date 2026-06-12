# AGENT.md — merlin/python/merlin/llvmlower

## Purpose

Whole-model lowering: linalg-on-tensors MLIR (model2MLIR artifacts) → upstream MLIR pipeline → LLVM IR → x86 (verification) / rv64gcv (deployment) objects. This is the llvm-project plane for running entire models (smolVLA) on RVV, complementing the per-kernel `runtime/backends` path.

## What belongs here

- `passes_xdsl.py` — Merlin-authored rewrites: `quant_ext.dequantize_per_channel` → `linalg.generic`; `llvm.emit_c_interface`; (future) `scf.parallel` → `merlin_parallel_for`.
- `pipeline.py` — upstream pass pipeline + `translate_module_to_llvmir` (in the model2MLIR venv).
- `codegen.py`, `toolchain.py`, `weights_pack.py` (manifest/safetensors → blob + arg table), `abi.py` (`_mlir_ciface_forward` host runner + `ScalarArg`), `lower.py`/`cli.py`.
- `kernel_backend.py` — compile one outlined kernel func in isolation + check it vs a numpy reference (the per-kernel bisection harness; used by `runtime.dispatch_runtime`).
- `custom_isa.py` — `merlin.inline_asm` → `llvm.inline_asm` 1:1 (custom ISA / `.insn` raw encodings; no LLVM fork). `passes_xdsl.lower_bf16_matmul_f32acc` rewrites bf16 matmuls to accumulate in f32.

## What does not belong here

- Hand-written kernels (`merlin/runtime/baremetal/spike/`), command-buffer pipeline (`xdsl_dialects/lowering/`), model capture (model2MLIR).

## Invariants

- **Target-agnostic.** Everything here is target-independent: the same `.ll` produces
  x86 (verification) and rv64gcv (deployment); the *only* place a target enters is
  clang's `--target`/`-march` in `codegen.py`, selected by `lower_model(..., target=)`.
  Do not branch on target in the passes, weights packer, ABI, or runner. Target-specific
  code belongs only in `merlin/runtime/baremetal/<env>/` and `xdsl_dialects/targets/`.
  The RVV vectorization stage (when added) must be a target-parameterized entry in the
  pass list, not a hardcoded fork.
- `buffer-results-to-out-params` MUST include `modify-public-functions hoist-static-allocs` — otherwise it silently skips the public `@forward` and the entry returns heap-allocated descriptors.
- `quant_ext.*` parses as `builtin.unregistered`: match `op.op_name.data`, not `op.name`.
- Weight tensors are never embedded in C arrays — pointers into the safetensors payload blob, offsets straight from the header (`weights_pack.pack`).
- Vectorization is clang `-O2 -march=rv64gcv` auto-vectorization (verified: emits vsetvli). A scalable-vector tile/vectorize MLIR path may be layered later.
- Host (x86 ctypes) parity vs torch reference is the gate before any spike run.
- `HostModel.load` defaults to `RTLD_LOCAL` (global only for the >1024-arg trampoline path) so several model/kernel `.so`s coexist in one process without their shared `forward`/`memrefCopy` symbols clashing. `emit_c_interface` wraps only memref args as descriptor pointers; scalar args are passed by value — use `abi.ScalarArg` (the dispatch runtime relies on this for `cumsum`-style kernels).

## Testing expectations

`merlin/python/tests/test_llvmlower.py` — synthetic slice e2e (host execution vs Python reference); toolchain-gated tests auto-skip when clang/m2m venv are absent.

## Notes for future agents

Tools: torch-mlir wheel python = full upstream pass registry + translate; clang-23 from `/scratch2/agustin/merlin/...` targets riscv64 with `+v`. The 27k-line full model goes through the venv pipeline as text — expect minutes, not seconds.
