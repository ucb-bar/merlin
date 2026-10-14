---
title: "Triton kernel integration — status, workflow and open work"
kind: guide
status: draft
owner: runtime
last_verified: 2026-10-14
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

1. **Kernel table** (`src/merlin/llvmlower/kernel_table.py`) — maps each `prov.kernel` name to one or
   more CPU kernel bodies (`src/merlin/kernels/triton_llvmir/`) and the C-level ABI each body expects.
2. **Link** (`src/merlin/llvmlower/kernel_link.py`) — during whole-model lowering, a generated C shim
   adapts the MLIR call (memref descriptors) to the body's ABI and loops over the Triton program grid;
   `llvm-link` merges shim + lowered model + body into one LLVM module.
3. **Target** — that module is compiled for rv64gcv and run on spike like any other model.

Until a backend dialect claims an external kernel, merlin's staged (dialect) passes leave the call
opaque and route the module to the LLVM path. Triton is only the authoring language: nothing runs on
a GPU, and the linked body is CPU code.

```
model.mlir (from model2MLIR)
  └─ find_kernel_uses ─────────────► KernelUse(symbol, prov.kernel, operand names, types)
  └─ preprocess + prepare_kernel_declarations (emit_c_interface, read-only args)
  └─ upstream MLIR pipeline ───────► model.ll        (calls _mlir_ciface_<sym>)
  └─ emit_shim ────────────────────► kernel_shim.c ─clang─► kernel_shim.ll
  └─ kernel body (table) ──────────► <entry>.<form>.link.ll   (llvm_mlir: mlir-translate first)
  └─ llvm-link shim + model + body ► model.linked.ll ─clang rv64gcv─► model.o ─► model.elf ─► spike
```

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

### 0. Prerequisites (once)

- `.env` with `MERLIN_CHIPYARD`, `MERLIN_SPIKE`, `MERLIN_RISCV_GCC` (chipyard's spike and
  `riscv64-unknown-elf` toolchain) and `MERLIN_CLANG` (LLVM 23 clang with the RISCV target) — see
  [rvv_e2e](rvv_e2e.md) and [llvm_toolchain](llvm_toolchain.md).
- `llvm-link` from the same LLVM build as that clang:
  ```bash
  .venv/bin/ninja -C third_party/llvm-build llvm-link
  ```
- spike shells out to `dtc`; put chipyard's env on `PATH` for any spike run:
  ```bash
  export PATH=<chipyard>/.conda-env/bin:$PATH
  ```
- Optional, only to regenerate the bundles: a model2MLIR checkout with the kernel registry
  (`MERLIN_M2M_DIR`, `MERLIN_M2M_VENV` in `.env`).

### 1. (Optional) regenerate the bundles from model2MLIR

```bash
$MERLIN_M2M_VENV/bin/python build_tools/scripts/gen_mlp_kernel_fixtures.py \
    --m2m $MERLIN_M2M_DIR --out merlin/tests/data/mlp_bundles
```

Captures both workloads fresh (`m2m.convert(..., backend="fx_importer")`) and computes the golden in
PyTorch. Deterministic; `merlin/tests/infra/test_mlp_bundles_fresh.py` fails when the tracked bundles
no longer match it.

### 2. Run on spike — both body forms

`triton_exp::fused_mlp2` has two bodies, both exported by triton-cpu, selected with `kernel_form`:

| `kernel_form` | body file | what it is |
|---|---|---|
| `llvm_ir` (default: the table's first) | `kernels/triton_llvmir/mlp2.ll` | textual LLVM IR, already optimized for the x86 host that exported it |
| `llvm_mlir` | `kernels/triton_llvmir/mlp2_llvm.mlir` | MLIR `llvm` dialect, before LLVM optimization; `mlir-translate`d, then optimized by clang for rv64gcv |

`build_tools/scripts/run_mlp_kernel_spike.py` copies a bundle to a work directory (the tracked one stays
clean), builds it with `spike_model.build(..., kernel_form=)`, runs it with `spike_model.run` and compares
the output with the bundle's golden:

```bash
.venv/bin/python build_tools/scripts/run_mlp_kernel_spike.py --form llvm_ir
.venv/bin/python build_tools/scripts/run_mlp_kernel_spike.py --form llvm_mlir
.venv/bin/python build_tools/scripts/run_mlp_kernel_spike.py --bundle mlp_basic     # the kernel-free twin
```

Expected (spike `rv64gcv_zfh_zvfh`, VLEN 128):

```
variant=mlp_basic_triton_llvm_ir kernels=['triton_exp::fused_mlp2:llvm_ir'] cycles=44358 max_abs_err=2.98e-08
variant=mlp_basic_triton_llvm_mlir kernels=['triton_exp::fused_mlp2:llvm_mlir'] cycles=15558 max_abs_err=2.98e-08
variant=mlp_basic kernels=[] cycles=16156 max_abs_err=4.47e-08
```

It exits non-zero if the error exceeds 1e-5. Without `--form` the table's first body (`llvm_ir`) is
linked. The work directory defaults to `out/artifacts/probes/mlp_kernel/<variant>/` (`--work` overrides).

Files left in `<work>/build/`:

| path | what |
|---|---|
| `lower/model.upstream.mlir` | the module handed to the upstream MLIR pipeline (kernel declaration normalized) |
| `lower/model.ll` | the lowered model alone; calls `_mlir_ciface_triton_exp__fused_mlp2` |
| `lower/kernel_shim.c`, `lower/kernel_shim.ll` | the generated adapter: descriptor checks, grid loop, result allocation |
| `lower/fused_mlp2_kernel.<form>.link.ll` | the body as linkable LLVM IR (exporter's triple dropped; for `llvm_mlir` also `fused_mlp2_kernel.llvm_mlir.ll`, the `mlir-translate` output) |
| `lower/model.linked.ll` | **the combined module** (shim + model + body) |
| `model.o` | `model.linked.ll` compiled for rv64gcv |
| `cgen/`, `*.o` | the data-driven C runtime, harness and weights blob |
| `model.elf` | the bare-metal image spike runs |

### 3. Lower only (no spike)

```python
from pathlib import Path
from merlin.common.paths import merlin_dir
from merlin.llvmlower.lower import lower_model

bundle = merlin_dir() / "tests" / "data" / "mlp_bundles" / "mlp_basic_triton"
res = lower_model((bundle / "model.mlir").read_text(), Path("out/artifacts/probes/mlp_kernel/lower"),
                  targets=("riscv", "host"), kernel_form="llvm_mlir")   # or "llvm_ir"
res.ll_path     # .../model.linked.ll
res.riscv_obj   # .../model_rv64gcv.o
res.host_so     # .../model_host.so  (x86, links its own copy of shim + body)
```

`res.host_so` can be called through `llvmlower.abi.HostModel` as an x86 oracle; both forms match
PyTorch there too.

### 4. Tests

```bash
PATH=<chipyard>/.conda-env/bin:$PATH .venv/bin/python -m pytest \
  merlin/tests/kernels/test_kernel_table.py merlin/tests/rvv/test_kernel_link.py \
  merlin/tests/ir/test_kernel_call_passes.py merlin/tests/runtime/test_external_kernel_spike.py \
  merlin/tests/runtime/test_dispatch_external_kernel.py merlin/tests/runtime/test_descriptor_layout.py \
  merlin/tests/gemmini/test_external_kernel_bench.py merlin/tests/infra/test_mlp_bundles_fresh.py
```

80 passed on 2026-10-14. Spike tests skip without spike/clang/`llvm-link`; the model2MLIR registry
cross-check and the freshness check skip without a model2MLIR checkout.

## Results

Spike `rv64gcv_zfh_zvfh`, VLEN 128, against the PyTorch eager golden.

| variant | max abs err | spike retired instructions | `.text` | kernel body size |
|---|---|---|---|---|
| `mlp_basic` (no kernel) | 4.5e-8 | 16,156 | 10.3 KB | — |
| `mlp_basic_triton`, `llvm_ir` body | 3.0e-8 | 44,358 | 150 KB | 141 KB |
| `mlp_basic_triton`, `llvm_mlir` body | 3.0e-8 | 15,558 | 46 KB | 37.6 KB |

- All three match PyTorch at fp32 rounding (relative error ~1e-6); the two body forms are bit-identical.
- spike's `cycles` counts retired instructions, not time. There are no FireSim/K1 timings yet.
- `llvm_ir` is 2.7× slower: its export was unrolled for x86 (`<32 x float>`), which legalizes into
  ~14.7k vector instructions on 128-bit RVV. **Prefer `llvm_mlir` for a CPU target.**
- The full measurement (ELFs, `results.json`, report) is under
  `out/artifacts/measurements/spike_rv64gcv/mlp_basic_triton/`.

## Changes made

Branch `feat/triton-integration2`, based on `main` `6f1acb6e6`.

**Kernel table and bodies**
- `src/merlin/llvmlower/kernel_table.py` (new): `KernelEntry` / `KernelBody` / `AbiParam`, keyed by
  `prov.kernel`. A body declares its form, IR file, entry symbol, ABI (`operand`, `result`, `dim`,
  `program_id`, `grid`, `null`), program-grid block, shape specialization and portability.
- `src/merlin/kernels/triton_llvmir/` (new, package data): `mlp2.ll`, `mlp2_llvm.mlir`, `AGENT.md`.

**Lowering and link**
- `src/merlin/llvmlower/kernel_link.py` (new): `find_kernel_uses`, `prepare_kernel_declarations`,
  `select_body`, `emit_shim`, `link_kernels`, and `_check_use`, which refuses unsupported uses by name
  before any code is generated.
- `src/merlin/llvmlower/lower.py`: `lower_model(..., kernel_form=)` links kernel bodies whenever the
  module declares one; kernel-free modules skip it after one string check and lower byte-identically.
- `src/merlin/runtime/backends/spike_model.py`: `build(..., kernel_form=)`, reports `kernels`.

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
- Updated: `docs/guides/rvv_e2e.md`, `docs/guides/llvm_toolchain.md`, `docs/guides/model2mlir.md`,
  `docs/design/triton_frontend.md`, `merlin/tests/data/AGENT.md`, and the `AGENT.md` of
  `src/merlin/{llvmlower,runtime,xdsl_dialects/lowering,kernels}/`.

## Limitations

Every gap below is **refused by name**; none produces a silently wrong answer.

| gap | what happens today | what it needs |
|---|---|---|
| Kernels that write into an argument (`mutates_args`) | model2MLIR rejects a registered one at import | destination-passing form in model2MLIR (`prov.kernel_mutated` / `prov.kernel_tied`); a tied-result `AbiParam` kind and no result allocation in the shim |
| More than one result per kernel | `_check_use` refuses | shim support for a struct-of-descriptors return |
| Non-tensor arguments (`prov.kernel_args`) | `_check_use` refuses | an `AbiParam` kind that binds a call-site attribute; model2MLIR keys symbols by shape only |
| Dynamic shapes | refused | — |
| Another shape of the same kernel | refused against `specialization` | bodies keyed by `(form, specialization)`; triton-cpu bakes K, H1, H2, BLOCK_M |
| Body export | not reproducible from this repo | the two bodies came from an external triton-cpu checkout and embed its absolute paths in debug info; add an export script or strip debug info on import |
| `llvm_ir` performance | 2.7× slower on RVV | export for riscv64, or use `llvm_mlir` |
| Other runners | `runtime.kernel_spike` links no allocator; host dispatch runtimes and `dse_guidance.dispatch_measure` refuse kernels | link a portable body into the host `.so` |
| Shim result ownership | relies on the model's deallocation freeing the shim's `malloc` | a repeated-inference leak/double-free check |
| merlin's own Triton frontend | cannot produce this kernel as linalg (a third body form) | transposed loads, `expand_dims` on loads, `arith.maxnumf`, `tt.dot` at grid > 1, runtime `M` |

## Current state

- **Branch.** `feat/triton-integration2` on ucb-bar/merlin (pushed) is the only copy of this work. No PR
  is open.
- **Verified.** Both body forms on spike against PyTorch (above); both on the x86 host oracle; the
  kernel table driven through each body's ABI on the host against numpy at M = 4, 16 and 20; both
  preprocess paths (pretty and generic printed declarations); the staged-path refusals.
- **Regression.** `kernels`, `rvv`, `runtime`, `infra`, `ir` and `targetgen` were compared against a
  clean `main` worktree; the one regression found (a duplicated `toolchain.llvm_link()`) was fixed.
  Remaining failures in this environment fail on `main` too (they need local `out/` artifacts or
  out-of-tree target packages).
- **Repo gates.** `check_format`, `check_structure`, `check_artifact_layout`, `check_docs`,
  `check_doc_paths`, `check_no_local_paths`, `check_no_regex`, `check_no_target_name` pass.
  `check_wiring` fails on clean `main` as well, so the pre-commit hook currently blocks every commit.
- **Not measured.** Time on real hardware (FireSim / K1) for any of the three images.
- **Known unrelated debt found along the way.** `src/merlin/kernels/ceiling_drivers/*.c` still writes
  descriptor strides field-wise (the bug `main` fixed in `merlin_model.c`); `build_tools/scripts/AGENT.md`
  describes its directory as gitignored although ~70 scripts are tracked.
- **Upstream dependency.** New captures need model2MLIR's kernel registry (branch
  `feat/triton-integration` on ucb-bar/model2MLIR). The tracked bundles let every test here run
  without it.
- **Next steps, in order of value.** (1) in-place kernels end to end; (2) `prov.kernel_args` and
  multiple results; (3) a reproducible body export (ideally triton-cpu targeting riscv64); (4)
  per-specialization bodies; (5) hardware timings.
