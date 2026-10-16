---
title: "Triton kernel integration — status, workflow and open work"
kind: guide
status: draft
owner: runtime
last_verified: 2026-10-15
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

> **Work in progress.** One kernel (`triton_exp::fused_mlp2`) on one toy model runs end to end on
> spike and matches PyTorch. The mechanism is general, but most kernel shapes beyond that one are
> refused today — see [Limitations](#limitations). The durable design is in
> [external_kernels](../design/external_kernels.md); this page is the status and how-to.

> **Temporary page.** It doubles as a status report and will be split up later to follow the docs
> convention: the how-to stays in `guides/`, rationale moves into `design/external_kernels.md`, and
> point-in-time status and results move under `out/artifacts/`.

## What it does

A model can call a hand-written kernel it does not define. model2MLIR captures a registered custom op
(today a Triton `@triton_op`) as a **black-box call**: a `func.call` to a body-less `func.func`, tagged
`prov.kernel = "<ns>::<name>"`. merlin supplies the kernel's code:

1. **Kernel table** (`src/merlin/llvmlower/kernel_table.py`) — maps each `prov.kernel` name to the
   `@triton.jit` kernel its `@triton_op` launches, exported by triton-cpu as MLIR in the `llvm` dialect
   (`src/merlin/kernels/triton_llvmir/`), and to how the wrapper launches it (argument order, grid).
2. **Link** (`src/merlin/llvmlower/kernel_link.py`) — before the upstream MLIR pipeline, the kernel
   declaration is replaced by the MLIR twin of the `@triton_op` wrapper (operand buffers, result
   allocation, one call per Triton program), and the kernel's `llvm.func` is merged into the module.
3. **Target** — the pipeline lowers that one module like any other model; it is compiled for rv64gcv
   and run on spike.

Until a backend dialect claims an external kernel, merlin's staged (dialect) passes leave the call
opaque and route the module to the LLVM path. Triton is only the authoring language: nothing runs on
a GPU, and the kernel is compiled for the model's CPU target.

```
model.mlir (from model2MLIR)
  └─ find_kernel_uses ───────────────► KernelUse(symbol, prov.kernel, operand names, types)
  └─ preprocess (xDSL | textual)
  └─ link_kernels ───────────────────► kernel_wrappers.mlir (generated) + kernel body (table)
                                        → model.upstream.mlir: one closed module
  └─ upstream MLIR pipeline ─────────► model.ll   (wrapper inlined into forward; kernel defined)
  └─ clang rv64gcv ──────────────────► model.o ─► model.elf ─► spike
```

There is no generated C, no second `mlir-translate`, and no `llvm-link`.

## End-to-end: the MLP toy example

Two bundles under `merlin/tests/data/mlp_bundles/`, captured by model2MLIR from the same 3-layer MLP
(16 → 32 → 32 → 8, same seed, weights and input):

| bundle | what |
|---|---|
| `mlp_basic/` | plain PyTorch; every op lowered to linalg |
| `mlp_basic_triton/` | fc1+relu+fc2+relu is one Triton op → one `func.call @triton_exp__fused_mlp2` |

Each holds `model.mlir`, `weights.safetensors` (+ `.manifest.json`), `inputs.npz`, `input_order.json`
and `golden.npy` (PyTorch eager output of `mlp_basic`, the oracle for both).

The kernel call merlin receives:

```mlir
%7 = func.call @triton_exp__fused_mlp2(%6, %0, %1, %2, %3) {prov.kernel = "triton_exp::fused_mlp2",
       prov.kernel_operands = ["x", "w1", "b1", "w2", "b2"], ...}
     : (tensor<4x16xf32>, tensor<32x16xf32>, tensor<32xf32>, tensor<32x32xf32>, tensor<32xf32>) -> tensor<4x32xf32>
...
func.func private @triton_exp__fused_mlp2(...) -> tensor<4x32xf32> attributes {prov.kernel = "triton_exp::fused_mlp2"}
```

The kernel body, `kernels/triton_llvmir/mlp2_wrapper_llvm.mlir`, is the `_mlp2_kernel` that model2MLIR's
`workloads/mlp_basic_triton/kernels.py` wrapper launches as
`_mlp2_kernel[(cdiv(M, 16),)](x, w1, b1, w2, b2, out, M, K=16, H1=32, H2=32, BLOCK_M=16)`.

### 0. Prerequisites (once)

- `.env` with `MERLIN_CHIPYARD`, `MERLIN_SPIKE`, `MERLIN_RISCV_GCC` (chipyard's spike and
  `riscv64-unknown-elf` toolchain) and `MERLIN_CLANG` (LLVM 23 clang with the RISCV target) — see
  [rvv_e2e](rvv_e2e.md) and [llvm_toolchain](llvm_toolchain.md).
- The compiler Python with the MLIR bindings (`MERLIN_M2M_VENV` or `MERLIN_COMPILER_VENV`), which every
  whole-model lowering already needs; the link runs in it too.
- spike shells out to `dtc`; put chipyard's env on `PATH` for any spike run:
  ```bash
  export PATH=<chipyard>/.conda-env/bin:$PATH
  ```
- Optional, only to regenerate the bundles: a model2MLIR checkout with the kernel registry
  (`MERLIN_M2M_DIR` in `.env`).

### 1. (Optional) regenerate the bundles from model2MLIR

```bash
$MERLIN_M2M_VENV/bin/python build_tools/scripts/gen_mlp_kernel_fixtures.py \
    --m2m $MERLIN_M2M_DIR --out merlin/tests/data/mlp_bundles
```

Captures both workloads fresh (`m2m.convert(..., backend="fx_importer")`) and computes the golden in
PyTorch. Deterministic; `merlin/tests/infra/test_mlp_bundles_fresh.py` fails when the tracked bundles
no longer match it.

### 2. Run on spike

`build_tools/scripts/run_mlp_kernel_spike.py` copies a bundle to a work directory (the tracked one stays
clean), builds it with `spike_model.build`, runs it with `spike_model.run` and compares the output with
the bundle's golden:

```bash
.venv/bin/python build_tools/scripts/run_mlp_kernel_spike.py                       # mlp_basic_triton
.venv/bin/python build_tools/scripts/run_mlp_kernel_spike.py --bundle mlp_basic    # the kernel-free twin
```

Expected (spike `rv64gcv_zfh_zvfh`, VLEN 128):

```
bundle=mlp_basic_triton kernels=['triton_exp::fused_mlp2'] cycles=15121 max_abs_err=2.98e-08
bundle=mlp_basic kernels=[] cycles=16156 max_abs_err=4.47e-08
```

It exits non-zero if the error exceeds 1e-5. The work directory defaults to
`out/artifacts/probes/mlp_kernel/<bundle>/` (`--work` overrides).

Files left in `<work>/build/`:

| path | what |
|---|---|
| `lower/model.unlinked.mlir` | the module as the preprocess left it, kernel declaration still body-less |
| `lower/kernel_wrappers.mlir` | the generated wrapper(s): operand buffers, result allocation, grid loop, `llvm.call` |
| `lower/kernel_link.py` | the merge that ran (in the compiler's Python) |
| `lower/model.upstream.mlir` | **the linked module** handed to the upstream MLIR pipeline: wrapper + kernel `llvm.func` |
| `lower/model.ll` | the lowered model, kernel included |
| `model.o` | `model.ll` compiled for rv64gcv |
| `cgen/`, `*.o` | the data-driven C runtime, harness and weights blob |
| `model.elf` | the bare-metal image spike runs |

### 3. Lower only (no spike)

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

`res.host_so` can be called through `llvmlower.abi.HostModel` as an x86 oracle; it matches PyTorch.

### 4. Tests

```bash
PATH=<chipyard>/.conda-env/bin:$PATH .venv/bin/python -m pytest \
  merlin/tests/kernels/test_kernel_table.py merlin/tests/rvv/test_kernel_link.py \
  merlin/tests/ir/test_kernel_call_passes.py merlin/tests/runtime/test_external_kernel_spike.py \
  merlin/tests/runtime/test_dispatch_external_kernel.py merlin/tests/runtime/test_descriptor_layout.py \
  merlin/tests/gemmini/test_external_kernel_bench.py merlin/tests/infra/test_mlp_bundles_fresh.py
```

73 passed on 2026-10-15. Spike tests skip without spike/clang; the model2MLIR registry cross-check and
the freshness check skip without a model2MLIR checkout.

## Results

Spike `rv64gcv_zfh_zvfh`, VLEN 128, against the PyTorch eager golden.

| variant | max abs err | spike retired instructions | `.text` | kernel code |
|---|---|---|---|---|
| `mlp_basic` (no kernel) | 4.5e-8 | 16,156 | 10.3 KB | — |
| `mlp_basic_triton`, MLIR-linked kernel | 3.0e-8 | **15,121** | 44.9 KB | 37.6 KB |
| `mlp_basic_triton`, C adapter + `llvm-link`, `llvm_mlir` body (before 2026-10-15) | 3.0e-8 | 15,558 | 46.2 KB | 37.6 KB |
| `mlp_basic_triton`, C adapter + `llvm-link`, `llvm_ir` body (before 2026-10-15) | 3.0e-8 | 44,358 | 150 KB | 141 KB |

- All match PyTorch at fp32 rounding (relative error ~1e-6). The MLIR-linked image and the old
  `llvm_mlir` image produce bit-identical outputs.
- spike's `cycles` counts retired instructions, not time. There are no FireSim/K1 timings yet.
- **Why 437 fewer than the old `llvm_mlir` build.** The kernel itself is unchanged: the export differs
  only in name, debug locations and `tt.divisibility`, and it compiles to the same 9,983 instructions
  and retires 12,297 either way. Every saved instruction is on the launch path (from a spike
  instruction log):
  - the C adapter is gone (113): no descriptor density checks, and no result descriptor;
  - the deallocation is static (`malloc` 212 → 67, `free` 7 → 2). The old body-less declaration
    *returned* its result buffer, so ownership-based deallocation could not prove it distinct from
    `forward`'s other allocation. It emitted a runtime alias check, a `dealloc_helper` call with five
    scratch heap arrays. Now the result is an out-param that `forward` allocates;
  - `forward` + `_mlir_ciface_forward` retire 2,345 instead of 2,519: `forward` no longer builds six
    memref descriptors for a C-interface call.
- The `llvm_ir` body is gone: its export was unrolled for x86 (`<32 x float>`), which legalizes into
  ~14.7k vector instructions on 128-bit RVV. Bodies are now accepted only in the `llvm` dialect,
  before LLVM optimization.
- The full measurement for the older rows (ELFs, `results.json`, report) is under
  `out/artifacts/measurements/spike_rv64gcv/mlp_basic_triton/`.

## Changes made

Branch `feat/triton-integration2`, based on `main` `6f1acb6e6`, and `trition-op-integration` on top of
it (local), which replaces the C adapter with the MLIR link.

**Kernel table and bodies**
- `src/merlin/llvmlower/kernel_table.py`: `KernelEntry` / `KernelBody` / `KernelArg`, keyed by
  `prov.kernel`. A body names its llvm-dialect MLIR file, its entry `llvm.func`, its arguments before
  the launch tail (`operand`, `result`, `dim`), the program grid and its shape specialization.
- `src/merlin/kernels/triton_llvmir/` (package data): `mlp2_wrapper_llvm.mlir`, `AGENT.md`. The
  triton-cpu CPU-twin exports `mlp2.ll` and `mlp2_llvm.mlir` were removed.

**Lowering and link**
- `src/merlin/llvmlower/kernel_link.py`: `find_kernel_uses`, `emit_wrappers`, `link_kernels`, and
  `_check_use`, which refuses unsupported uses by name before any code is generated.
- `src/merlin/llvmlower/lower.py`: `lower_model` links kernel bodies whenever the module declares one;
  kernel-free modules skip it after one string check and lower byte-identically.
- `src/merlin/runtime/backends/spike_model.py`: `build` reports `kernels`.

**Staged path and runtimes** (until a backend dialect claims a kernel)
- `src/merlin/common/kernel_calls.py` (new): `is_kernel_call`, `kernel_qualname`, `is_external_decl`, `entry_function`, `ExternalKernelNotExecutable`.
- Outliner keeps the call and carries the declaration; the dispatch program records a
  `kind="external"` node; `support_ops` no longer absorbs the call (it would have deleted two layers);
  `lower_to_interface` refuses it; `compile_core.choose_route` sends it to the LLVM route; the host
  dispatch runtimes raise `ExternalKernelNotExecutable`.

**Test data and tests**
- `merlin/tests/data/mlp_bundles/` + `build_tools/scripts/gen_mlp_kernel_fixtures.py` (fresh capture,
  deterministic) + `merlin/tests/fixtures/mlp_bundles.py` (loader helper).
- `build_tools/scripts/run_mlp_kernel_spike.py`: the end-to-end example above.
- Tests: `kernels/test_kernel_table.py`, `rvv/test_kernel_link.py`, `ir/test_kernel_call_passes.py`,
  `runtime/test_external_kernel_spike.py`, `runtime/test_dispatch_external_kernel.py`,
  `runtime/test_descriptor_layout.py`, `infra/test_mlp_bundles_fresh.py`.

**Docs**
- New: `docs/design/external_kernels.md`, `src/merlin/kernels/triton_llvmir/AGENT.md`,
  `merlin/tests/data/mlp_bundles/AGENT.md`, this page.
- Updated: `docs/guides/rvv_e2e.md`, `docs/guides/model2mlir.md`, `docs/design/triton_frontend.md`,
  `merlin/tests/data/AGENT.md`, and the `AGENT.md` of
  `src/merlin/{llvmlower,runtime,xdsl_dialects/lowering,kernels}/`.

## Limitations

Every gap below is **refused by name**; none produces a silently wrong answer.

| gap | what happens today | what it needs |
|---|---|---|
| Kernels that write into an argument (`mutates_args`) | model2MLIR rejects a registered one at import | destination-passing form in model2MLIR (`prov.kernel_mutated` / `prov.kernel_tied`); a tied `KernelArg` kind whose buffer the wrapper takes writable and returns instead of allocating |
| More than one result per kernel | the wrapper supports it; no registered kernel has one | a kernel to exercise it |
| Non-tensor arguments (`prov.kernel_args`) | `_check_use` refuses | a `KernelArg` kind that binds a call-site attribute; model2MLIR keys symbols by shape only |
| Dynamic shapes | refused | — |
| Another shape of the same kernel | refused against `specialization` | bodies keyed by specialization; triton-cpu bakes K, H1, H2, BLOCK_M |
| Body export | not reproducible from this repo | an export script. The body's debug paths no longer reach the image (the link drops locations) |
| LLVM IR (`.ll`) bodies | not accepted | `mlir-translate --import-llvm` first; the one host-optimized export was 2.9× slower on RVV |
| `tt.divisibility` | recorded, not checked | a wrapper check (or blob padding) for a body that relies on 16-byte alignment; `mlp2` needs only 4 |
| Other runners | `runtime.kernel_spike` links no allocator; host dispatch runtimes and `dse_guidance.dispatch_measure` refuse kernels | route them through `lower_model`, which builds the host `.so` with the kernel already inside |
| merlin's own Triton frontend | cannot produce this kernel as linalg | transposed loads, `expand_dims` on loads, `arith.maxnumf`, `tt.dot` at grid > 1, runtime `M` |

## Current state

- **Branch.** `feat/triton-integration2` on ucb-bar/merlin (pushed) holds the C-adapter version. The
  local branch `trition-op-integration` on top of it holds the MLIR link. No PR is open.
- **Verified.** The kernel on spike against PyTorch (above). On the x86 host oracle: the whole model
  against the golden through both preprocess paths; the kernel table's launch at M = 4, 16 and 20 against
  numpy; and a strided operand (every other row of `x`) reaching the kernel dense. Also the staged-path
  refusals.
- **Repo gates.** Run the pre-commit hook by hand (it is not installed in every clone). `check_wiring`
  fails on clean `main` as well.
- **Not measured.** Time on real hardware (FireSim / K1).
- **Known unrelated debt found along the way.** `src/merlin/kernels/ceiling_drivers/*.c` still writes
  descriptor strides field-wise (the bug `main` fixed in `merlin_model.c`); `build_tools/scripts/AGENT.md`
  describes its directory as gitignored although ~70 scripts are tracked.
- **Upstream dependency.** New captures need model2MLIR's kernel registry (branch
  `feat/triton-integration` on ucb-bar/model2MLIR). The tracked bundles let every test here run
  without it.
- **Next steps, in order of value.** (1) in-place kernels end to end; (2) `prov.kernel_args`; (3) a
  reproducible body export (ideally triton-cpu targeting riscv64); (4) per-specialization bodies; (5)
  hardware timings.
