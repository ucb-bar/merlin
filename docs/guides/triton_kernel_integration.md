---
title: "Triton kernel integration — how it works, its limits, and how to use it"
kind: guide
status: current
owner: runtime
last_verified: 2026-11-02
related: [external_kernels, rvv_e2e, llvm_toolchain, model2mlir, triton_frontend]
code_refs:
  - src/merlin/llvmlower/kernel_table.py
  - src/merlin/llvmlower/kernel_link.py
  - src/merlin/llvmlower/lower.py
  - src/merlin/runtime/backends/spike_model.py
  - src/merlin/common/kernel_calls.py
  - build_tools/scripts/gen_mlp_kernel_fixtures.py
  - build_tools/scripts/run_mlp_kernel_spike.py
---

# Triton kernel integration (merlin side)

A model can call a hand-written kernel it does not define, such as a Triton `@triton_op`.
model2MLIR captures each call to a registered kernel as a black-box `func.call` tagged
`prov.kernel = "<ns>::<name>"`. merlin supplies the kernel's code when it lowers the model, compiles
it for the model's target together with the model, and runs it on spike.

This page covers the lowering side. The capture side (the registry, the IR contract and how to
register a kernel) is model2MLIR's
[Triton kernel integration guide](https://github.com/ucb-bar/model2MLIR/blob/feat/triton-integration/docs/guides/triton_kernel_integration.md)
(`docs/guides/triton_kernel_integration.md` on model2MLIR's `feat/triton-integration` branch). The
design rationale is in [external_kernels](../design/external_kernels.md).

- [Part 1: Summary](#part-1-summary): [what it does](#what-it-does), [how it is implemented](#how-it-is-implemented), [results](#results), [limitations](#limitations)
- [Part 2: How to use it](#part-2-how-to-use-it): [prerequisites](#0-prerequisites-once), [regenerate](#1-optional-regenerate-the-bundles-from-model2mlir), [run on spike](#2-run-on-spike), [lower only](#3-lower-only-no-spike), [add a kernel](#4-add-a-kernel), [tests](#5-tests)

---

# Part 1: Summary

## What it does

Given a model2MLIR capture that calls a registered kernel, merlin:

1. finds each `prov.kernel` declaration and checks the call against what the kernel's body can
   compute: shapes, dtypes and operand names. Anything else is refused by name before any code is
   generated;
2. replaces the declaration with the MLIR twin of the `@triton_op` Python wrapper and merges the
   kernel's `llvm.func` into the model;
3. lowers that one closed module through the normal upstream pipeline, compiles it with clang for
   rv64gcv, and runs it on spike. The same `model.ll` also builds an x86 host `.so` that can serve as an
   oracle.

Triton is only the authoring language. Nothing runs on a GPU, and the kernel is compiled for the
model's CPU target. Kernel-free models skip all of this after one string check and lower
byte-identically.

## How it is implemented

```
model.mlir (from model2MLIR)
  └─ find_kernel_uses ───────────────► KernelUse(symbol, prov.kernel, operand names, types)
  └─ preprocess (xDSL | textual)
  └─ link_kernels ───────────────────► kernel_wrappers.mlir (generated) + kernel body (from the table)
                                        → model.upstream.mlir: one closed module
  └─ upstream MLIR pipeline ─────────► model.ll   (wrapper inlined into forward; kernel defined)
  └─ clang rv64gcv ──────────────────► model.o ─► model.elf ─► spike
```

| piece | file | what it does |
|---|---|---|
| Kernel table | `src/merlin/llvmlower/kernel_table.py` | `KERNEL_TABLE`, keyed by `prov.kernel` and never by symbol. Each `KernelEntry` has a `KernelBody`: the llvm-dialect MLIR file, its entry `llvm.func`, the `args` before triton-cpu's six launch integers (`operand`, `result`, `dim`), the program `grid`, and the shape `specialization` that Triton's `tl.constexpr` arguments baked in. |
| Kernel bodies | `src/merlin/kernels/triton_llvmir/` | `mlp2_wrapper_llvm.mlir`: the `@triton.jit` kernel `_mlp2_kernel`, exported by triton-cpu in the `llvm` dialect before LLVM optimization. It must be target-neutral: no arch intrinsics, inline asm or address spaces. |
| Link | `src/merlin/llvmlower/kernel_link.py` | `find_kernel_uses` reads the declarations and their call sites. `_check_use` refuses what the body cannot compute. `emit_wrappers` builds one `func.func` per use: read-only operand buffers, an allocated result, an `scf` grid loop and one `llvm.call` per program. `link_kernels` swaps the declarations for these wrappers and moves the body in. The merge is structural, done in the compiler's MLIR Python bindings, and printed without locations. |
| Lowering | `src/merlin/llvmlower/lower.py` | `lower_model` links whenever the module declares a kernel, after either preprocess path, and reports `stats["kernels_linked"]`. |
| spike runner | `src/merlin/runtime/backends/spike_model.py` | `build` reports `kernels`; `run` returns outputs and metrics. |
| Staged path | `src/merlin/common/kernel_calls.py` and the dialect passes | Until a backend dialect claims an external kernel, the staged passes keep the call opaque. The outliner carries the declaration, the dispatch program records a `kind="external"` node, `compile_core.choose_route` sends the model to the LLVM route, and the host dispatch runtimes raise `ExternalKernelNotExecutable`. |
| Test data | `merlin/tests/data/mlp_bundles/`, `build_tools/scripts/gen_mlp_kernel_fixtures.py` | Fresh, deterministic model2MLIR captures of `mlp_basic` and `mlp_basic_triton`, each with weights, inputs and a PyTorch golden. |

Because the wrapper is ordinary MLIR in the same module, the upstream pipeline does what a
hand-written C adapter used to do:

- one-shot-bufferize sees that the operands are only read, so no weight is copied before the call;
- `buffer-results-to-out-params{hoist-static-allocs}` hoists the result allocation into `@forward`, so
  the kernel writes straight into the caller's buffer and the model's deallocation frees it;
- the wrapper takes identity-layout memrefs, so bufferization copies any strided operand dense before
  it reaches the raw-pointer kernel.

There is no generated C, no second `mlir-translate`, and no `llvm-link`.

## Results

spike `rv64gcv_zfh_zvfh`, VLEN 128, against the PyTorch eager golden:

| variant | max abs error | spike retired instructions | `.text` | kernel code |
|---|---|---|---|---|
| `mlp_basic` (no kernel) | 4.5e-8 | 16,156 | 10.3 KB | — |
| `mlp_basic_triton` (kernel linked as MLIR) | 3.0e-8 | **15,121** | 44.9 KB | 37.6 KB |

Both match PyTorch at fp32 rounding, with a relative error of about 1e-6. spike's `cycles` counts
retired instructions, not time; there are no FireSim or K1 timings yet.

An earlier version called the kernel through a generated C adapter and `llvm-link`, and retired
15,558. The kernel code is the same in both: it retires 12,297 either way, and the outputs are
bit-identical. All 437 saved instructions are on the launch path:

- removing the adapter's descriptor checks saves 113;
- deallocation is now static (`malloc` 212 → 67, `free` 7 → 2). The old body-less callee returned its
  buffer, which forced a runtime alias check and a `dealloc_helper` call with five scratch arrays.

An `.ll` body exported for x86 legalized into about 14.7k vector instructions on 128-bit RVV and
retired 44,358. That is why bodies are now accepted only as llvm-dialect MLIR, before LLVM optimization.

## Limitations

Every gap below is **refused by name**; none produces a silently wrong answer.

| gap | what happens today | what it needs |
|---|---|---|
| Kernels that write into an argument (`mutates_args`) | model2MLIR rejects a registered one at import | A destination-passing form in model2MLIR (`prov.kernel_mutated` / `prov.kernel_tied`), and a tied `KernelArg` kind whose buffer the wrapper takes writable and returns instead of allocating |
| More than one result per kernel | The wrapper supports it, but no registered kernel has one | A kernel to exercise it |
| Non-tensor arguments (`prov.kernel_args`) | `_check_use` refuses | A `KernelArg` kind that binds a call-site attribute. model2MLIR also keys symbols by shape only. |
| Dynamic shapes | Refused | — |
| Another shape of the same kernel | Refused against `specialization` | Bodies keyed by specialization, because triton-cpu bakes in K, H1, H2 and BLOCK_M |
| Body export | Not reproducible from this repo | An export script, ideally triton-cpu targeting riscv64 |
| LLVM IR (`.ll`) bodies | Not accepted | `mlir-translate --import-llvm` first |
| `tt.divisibility` | Recorded but not checked | A wrapper check (or blob padding) for a body that relies on 16-byte alignment. `mlp2` needs only 4. |
| Other runners | `runtime.kernel_spike` links no allocator, and the host dispatch runtimes and `dse_guidance.dispatch_measure` refuse kernels | Route them through `lower_model`, which builds the host `.so` with the kernel already inside |
| merlin's own Triton frontend | Cannot produce this kernel as linalg | Transposed loads, `expand_dims` on loads, `arith.maxnumf`, `tt.dot` at grid > 1, and runtime `M` |

---

# Part 2: How to use it

The example is two bundles under `merlin/tests/data/mlp_bundles/`, captured by model2MLIR from the
same 3-layer MLP (16 → 32 → 32 → 8, same seed, weights and input):

| bundle | what |
|---|---|
| `mlp_basic/` | plain PyTorch; every op lowered to linalg |
| `mlp_basic_triton/` | fc1+relu+fc2+relu is one Triton op, captured as one `func.call @triton_exp__fused_mlp2` |

Each holds `model.mlir`, `weights.safetensors` (+ `.manifest.json`), `inputs.npz`, `input_order.json`
and `golden.npy`, the PyTorch eager output of `mlp_basic` and the oracle for both. The call merlin
receives:

```mlir
%7 = func.call @triton_exp__fused_mlp2(%6, %0, %1, %2, %3) {prov.kernel = "triton_exp::fused_mlp2",
       prov.kernel_operands = ["x", "w1", "b1", "w2", "b2"], ...}
     : (tensor<4x16xf32>, tensor<32x16xf32>, tensor<32xf32>, tensor<32x32xf32>, tensor<32xf32>) -> tensor<4x32xf32>
...
func.func private @triton_exp__fused_mlp2(...) -> tensor<4x32xf32> attributes {prov.kernel = "triton_exp::fused_mlp2"}
```

## 0. Prerequisites (once)

- A `.env` with `MERLIN_CHIPYARD`, `MERLIN_SPIKE` and `MERLIN_RISCV_GCC` (chipyard's spike and the
  `riscv64-unknown-elf` toolchain), and `MERLIN_CLANG` (LLVM 23 clang with the RISCV target). See
  [rvv_e2e](rvv_e2e.md) and [llvm_toolchain](llvm_toolchain.md).
- The compiler Python with the MLIR bindings (`MERLIN_M2M_VENV` or `MERLIN_COMPILER_VENV`). Every
  whole-model lowering already needs it, and the link runs in it too.
- spike shells out to `dtc`, so put chipyard's env on `PATH` for any spike run:
  ```bash
  export PATH=<chipyard>/.conda-env/bin:$PATH
  ```
- Optional, only to regenerate the bundles: a model2MLIR checkout with the kernel registry, set as
  `MERLIN_M2M_DIR` in `.env`.

## 1. (Optional) regenerate the bundles from model2MLIR

```bash
$MERLIN_M2M_VENV/bin/python build_tools/scripts/gen_mlp_kernel_fixtures.py \
    --m2m $MERLIN_M2M_DIR --out merlin/tests/data/mlp_bundles
```

This captures both workloads fresh (`m2m.convert(..., backend="fx_importer")`) and computes the golden
in PyTorch. It is deterministic, and `merlin/tests/infra/test_mlp_bundles_fresh.py` fails when the
tracked bundles no longer match it.

## 2. Run on spike

`build_tools/scripts/run_mlp_kernel_spike.py` copies a bundle to a work directory, so the tracked one
stays clean. It then builds the copy with `spike_model.build`, runs it with `spike_model.run`, and
compares the output with the bundle's golden:

```bash
.venv/bin/python build_tools/scripts/run_mlp_kernel_spike.py                       # mlp_basic_triton
.venv/bin/python build_tools/scripts/run_mlp_kernel_spike.py --bundle mlp_basic    # the kernel-free twin
```

Expected output:

```
bundle=mlp_basic_triton kernels=['triton_exp::fused_mlp2'] cycles=15121 max_abs_err=2.98e-08
bundle=mlp_basic kernels=[] cycles=16156 max_abs_err=4.47e-08
```

The script exits non-zero if the error exceeds 1e-5. The work directory defaults to
`out/artifacts/probes/mlp_kernel/<bundle>/`; `--work` overrides it.

Files left in `<work>/build/`:

| path | what |
|---|---|
| `lower/model.unlinked.mlir` | the module as the preprocess left it, with the kernel declaration still body-less |
| `lower/kernel_wrappers.mlir` | the generated wrapper(s): operand buffers, result allocation, grid loop, `llvm.call` |
| `lower/kernel_link.py` | the merge that ran, in the compiler's Python |
| `lower/model.upstream.mlir` | **the linked module** handed to the upstream MLIR pipeline: wrapper + kernel `llvm.func` |
| `lower/model.ll` | the lowered model, kernel included |
| `model.o` | `model.ll` compiled for rv64gcv |
| `cgen/`, `*.o` | the data-driven C runtime, harness and weights blob |
| `model.elf` | the bare-metal image spike runs |

## 3. Lower only (no spike)

```python
from pathlib import Path
from merlin.common.paths import merlin_dir
from merlin.llvmlower.lower import lower_model

bundle = merlin_dir() / "tests" / "data" / "mlp_bundles" / "mlp_basic_triton"
res = lower_model((bundle / "model.mlir").read_text(), Path("out/artifacts/probes/mlp_kernel/lower"),
                  targets=("riscv", "host"))
res.ll_path                    # .../model.ll (kernel included)
res.stats["kernels_linked"]    # ['triton_exp::fused_mlp2']
res.riscv_obj                  # .../model_rv64gcv.o
res.host_so                    # .../model_host.so  (x86, from the same model.ll)
```

`res.host_so` can be called through `llvmlower.abi.HostModel` as an x86 oracle, and it matches PyTorch.

## 4. Add a kernel

1. Register the op in model2MLIR. See
   [its guide](https://github.com/ucb-bar/model2MLIR/blob/feat/triton-integration/docs/guides/triton_kernel_integration.md#4-register-a-new-kernel).
2. Export the `@triton.jit` kernel that the `@triton_op` launches with triton-cpu, as llvm-dialect MLIR
   before LLVM optimization. Put the file under `src/merlin/kernels/triton_llvmir/` and keep it
   byte-identical to the export.
3. Add a `KernelEntry` to `_BUILTIN` in `src/merlin/llvmlower/kernel_table.py`. Its `KernelBody`
   transcribes the wrapper's launch: `args` in the `@triton.jit` signature's order, `grid` as
   `(operand, axis, block)` per program axis (the wrapper's `triton.cdiv`), and `specialization` as the
   shapes the `tl.constexpr` arguments were compiled for. A wrong argument list fails MLIR verification
   at link time.
4. Check the launch on the host against numpy in `merlin/tests/kernels/test_kernel_table.py`.
   `test_table_keys_match_m2m_registry` there fails until the two tables agree.

## 5. Tests

```bash
PATH=<chipyard>/.conda-env/bin:$PATH .venv/bin/python -m pytest \
  merlin/tests/kernels/test_kernel_table.py merlin/tests/rvv/test_kernel_link.py \
  merlin/tests/ir/test_kernel_call_passes.py merlin/tests/runtime/test_external_kernel_spike.py \
  merlin/tests/runtime/test_dispatch_external_kernel.py merlin/tests/runtime/test_descriptor_layout.py \
  merlin/tests/gemmini/test_external_kernel_bench.py merlin/tests/infra/test_mlp_bundles_fresh.py
```

All 73 pass with spike, clang and a model2MLIR checkout available. The spike tests skip without
spike or clang. The model2MLIR registry cross-check and the freshness check skip without a model2MLIR
checkout.
