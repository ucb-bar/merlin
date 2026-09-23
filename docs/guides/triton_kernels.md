---
title: "Compiling Triton kernels with Merlin"
kind: guide
status: draft
owner: ir
last_verified: 2026-09-21
related: [triton_frontend, lowering_pipeline, target_resolution]
code_refs:
  - src/merlin/triton/cli.py
  - src/merlin/triton/spec.py
  - src/merlin/triton/bridge.py
  - src/merlin/compile_core.py
---

# Compiling Triton kernels with Merlin

A standard `@triton.jit` kernel is a legal Merlin input. You do not write anything target-specific,
and Merlin does not generate anything Triton-specific: the kernel enters *above* the point where a
target is chosen, so a target that Merlin can already compile generic computation for is
Triton-programmable with no extra work. The design and its invariants are in
[Triton as a target-independent kernel frontend](../design/triton_frontend.md).

## Install

Triton is an optional extra, exact-pinned because the frontend drives compiler internals that carry
no stability promise:

```bash
uv pip install -e '.[triton]'
```

`merlin.triton.toolchain.probe()` reports what is installed and why it is or is not usable. It also
detects a *stripped* install — a Python-only tree with no `triton/_C/libtriton` — which otherwise
reports a plausible version and then fails deep inside the compiler.

## Compile one

```bash
merlin-compile-kernel examples/triton/vector_add.py:vector_add --target saturn \
    --arg 'x_ptr=*fp32:1025:read' --arg 'y_ptr=*fp32:1025:read' \
    --arg 'out_ptr=*fp32:1025:write' --arg 'n_elements=i32' \
    --assume n_elements=1025 --constexpr BLOCK_SIZE=256 --grid 5 \
    --emit all --verify
```

Artifacts land in a versioned product directory under `out/artifacts/triton-kernel/<target>/v0/`
(override with `--out`): the TTIR, the core MLIR, each staged module, the command buffer, and a
report carrying the route, the Triton version, the TTIR digest and the capability breakdown.

## What you have to declare, and why none of it is inferred

A Triton kernel is not self-describing. Its parameters are untyped, its pointers carry no shape, its
grid lives at the call site, and which buffers it writes is visible only inside the body. Merlin asks
for those rather than guessing, because a wrong guess here is a miscompile, not an error.

| flag | what it states |
| --- | --- |
| `--arg NAME=*DTYPE:SHAPE:EFFECT` | a pointer: element type, static shape, and `read` / `write` / `readwrite` |
| `--arg NAME=DTYPE` | a scalar parameter |
| `--constexpr NAME=VALUE` | a `tl.constexpr`; `BLOCK_*` / `GROUP_*` are portable meta-parameters |
| `--assume NAME=VALUE` | the compile-time value of a *runtime* scalar |
| `--grid X[,Y[,Z]]` | the SPMD launch grid |

`--assume` is the one that surprises people. In vector add the extent arrives as a runtime scalar, so
nothing in the kernel says `n_elements` equals the declared shape — and without that, the compiler
cannot check that the mask keeps the launch inside the tensor. It refuses rather than assuming.

**Effects are declared, not discovered.** A kernel that mutates a buffer the caller believes is
read-only is a miscompile, so the compiler cross-checks your declaration against what the kernel
actually does and rejects a disagreement in either direction.

**`--num-warps` / `--num-stages` are recorded, never interpreted.** They are CUDA scheduling knobs;
treating them as portable target semantics would be a lie. They appear in the report as provenance.

## Where a kernel goes, and why

The route is chosen by the **payload**, not by the target:

- matmul-family payload, on a target whose dialect plan covers it → the staged pipeline: contract →
  schedule → interface → the target's own dialect → runtime → command buffer;
- anything else → the generic LLVM path, **even on an accelerator**.

So a vector add compiles as generic computation on a systolic array, and a matmul takes the staged
path even on a CPU-class target. `--route-only` prints the decision, with the target's declared
coverage and what the interface layer can actually build, and stops.

## What is supported today, and what a refusal means

The bridge re-raises pointer arithmetic and the SPMD grid back into whole tensors. It accepts a
kernel when the accesses of every program instance tile each declared argument **exactly** — every
element once, in order. Masked tails are handled and *checked*: the mask is what stops
`ceil(N/BLOCK)` instances from running past the end, so the same kernel is accepted at `n=1024` and
refused at `n=1000` if its bounds check is missing.

Refusals name the boundary they hit rather than failing generically:

| refusal | meaning |
| --- | --- |
| `no translation for tt.<op>` | that Triton op has no lowering yet |
| `not covered exactly by the launch` | the grid, shape and mask do not agree — often a missing mask |
| `accessed in full but not in order` | a permutation such as a transposed tile |
| `index expression is not affine` | the addressing needs a real polyhedral analysis |
| `runtime scalar ... has no compile-time value` | add `--assume` |
| `tt.dot under a grid of N programs` | a contraction per program is not a whole-tensor contraction |
| `would silently drop N op(s) of the payload` | the staged pipeline cannot carry this shape; nothing was compiled |

Not yet supported: multi-program tiled GEMM with cross-program accumulation, masked tails on the
staged accelerator path, reductions and softmax, and atomics. Each fails closed.

## Audit intermediate lowering IR

Add `--ir-audit [exact|compact|both]` to a normal `merlin-compile-kernel` invocation (or
`python -m merlin.triton.cli`). It retains exact named lowering stages and
`index.json` with their order, hashes, and completion/failure under the selected
`--out` directory: `ir-audit-*/` for staged lowering, `llvm/ir-audit-*/` for LLVM.
The option is independent of `--emit`, which selects the ordinary output files.
`--route-only` still stops without writing anything. A bare `--ir-audit` means `exact`.

Compact views cover the staged xDSL route's six named stages and native torch-MLIR
parsed/lowered operations and pass-manager snapshots. Whole-model structural xDSL
preprocessing additionally records parsed IR, dead-tensor pruning, quantization-extension
lowering and C-interface annotation. The textual preprocessing variant still records
only its input/output stages. Other Python rewrites and
unsupported printers still produce hash-only records, not readable views. `both`
retains exact named-stage snapshots as well as supported compact views. `compact` keeps exact
parent hashes but does not copy exact IR into the audit directory. Ordinary executable
build artifacts still exist and are not shrunk or replaced.

Native views use `.inspection.mlir.txt`, carry a `NOT EXECUTABLE IR` header, and use
the printer's 64-element / 64-resource-character limits. These limits apply per attribute,
not to total module size. The parent hash covers native `get_asm()` serialization,
which can differ from the incoming source spelling. No compact view is fed back to
the compiler, and elision does not externalize weights to safetensors.

xDSL views use generic operation syntax, bypassing custom operation printers, and
elide dense integer/float tensors above 64 elements. Their stage-local `inspection`
records in `index.json` bind the view to the exact standard xDSL serialization hash.
Small tensors remain readable; other attribute kinds are not elided. Neither this
diagnostic printer nor native inspection modifies the live module or executable output.

For existing external tensor payloads, repeat `--audit-sidecar PATH` for each
declared weights file, argument manifest, and other required payload. This requires
`--ir-audit` and refuses missing files. It binds bytes in place; it does **not**
externalize embedded constants or copy a self-contained model bundle. Existing
capture safetensors/manifest and lifted or quantized inner-tensor payloads remain
the source of truth.

Compact/both xDSL stage views now reference separate raw tensor payloads for dense
attributes above 64 elements. Each stage's `inspection.tensors` records MLIR type,
shape, bytes and hash; identical bytes are deduplicated within the audit under
`tensors/<sha256>.bin`. No numeric conversion occurs. These inspection payloads
are not safetensors or executable replacements; native pass-printer views still
elide constants without exporting their tensor bytes.

Auditing defaults off because full IR snapshots can contain large dense tensors.
These are named-stage observations, not every internal compiler pass. See
[the compiler-plane inspection contract](../design/compiler_plane.md#inspecting-named-lowering-stages).

## Verify it yourself

```bash
.venv/bin/python -m pytest merlin/tests/ir/test_triton_bridge.py \
    merlin/tests/ir/test_triton_host_e2e.py merlin/tests/ir/test_triton_convergence.py -q
.venv/bin/python -m pytest merlin/tests/rvv/test_triton_rvv_e2e.py -q          # spike
.venv/bin/python -m pytest merlin/tests/gemmini/test_triton_gemmini_c0.py -q   # spike + verilator
```
