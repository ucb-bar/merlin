---
title: Design note: the future MLIR/C++ compiler plane
kind: design
status: current
owner: core
last_verified: 2026-09-21
related: [architecture]
code_refs: [src/merlin/xdsl_dialects, src/merlin/common/ir_audit.py, src/merlin/compile_core.py, src/merlin/llvmlower/lower.py]
---

# Design note: the (future) stable MLIR/C++ compiler plane

**Status: not built. The active compiler plane is Python + xDSL** under
`src/merlin/xdsl_dialects/` (five dialects: contract, schedule, interface, runtime, dse)
and `src/merlin/{targetgen,llvmlower}`.

The original design reserved a `merlin/compiler/` tree for an *eventual* stabilized C++/TableGen
plane — durable dialect definitions, lowering passes, and `merlin-opt`/`merlin-translate` tools —
into which an xDSL prototype would be "promoted" once it stabilized. That tree was pure scaffold
(AGENT.md placeholders, zero `.td`/`.cpp`/CMake) and was removed to keep the repo stub-free.

**If/when a C++ plane is actually needed:** create `merlin/compiler/` then with real TableGen +
CMake, promoting a specific xDSL dialect that has stabilized — don't reintroduce an empty skeleton.
Keep experimental analysis in Python; the C++ plane is only for stabilized, performance-critical
dialects/passes.

## Inspecting named lowering stages

`compile_core_mlir(module, ..., workdir=run_workdir, ir_audit=True)` retains exact
named-stage IR under a fresh `run_workdir/ir-audit-*/` directory. The same option is
available on `xdsl_dialects.lowering.lower_module` and
`llvmlower.lower.lower_model` / `lower_model_file`. It defaults to **off**: arbitrary
dense constants can make full snapshots large. The in-memory staged API writes nothing
without an explicit workdir, even when the option is enabled.

`index.json` records ordered stage files, their byte counts and SHA-256 digests,
the producing owner source, and completion or failure. Completed stage prefixes survive
a later lowering failure. The LLVM subprocess boundary additionally records actual
launch arguments (including its pass pipeline), observed executable/generated-runner
bytes, and the existing toolchain version observations. This is not a full dependency
closure or proof of concurrent immutability. No passes run on archived copies, and
auditing neither splits nor reorders passes. A failure to write requested evidence
refuses the invocation; an already-active lowering exception remains the primary error.

The staged path records input, contract, schedule, interface, target, and runtime IR.
The LLVM path records input, preprocessed upstream, scheduled upstream, translated or
LLVM-dialect output, normalized LLVM IR, and final LLVM IR. Native `compact` and `both`
audits additionally enable the MLIR pass manager's before/after printer without changing
pass scheduling. `passes/segment-NNNN/` retains native operation-scoped dumps in the
printer's directory layout; `index.json#pass_inspection` records each segment's pipeline,
file hashes and byte counts, including prefixes from failed lowering. These are compact
inspection-only views (64-element/resource limits), even though the native printer uses
`.mlir` filenames. Never use them as executable compiler inputs. They cover native MLIR
passes, not Python rewrites or downstream LLVM/clang passes; `both` retains exact named
stages, not exact copies after every pass. `merlin-opt --print-between-passes` remains
the existing stdout-only inspection option for that separate compiler entrypoint.

Pass existing captured weights and manifests with
`audit_sidecars=(weights_path, manifest_path)`. The index binds their observed bytes
in place and refuses missing files or observed end-of-invocation changes. It does not
copy, convert, or silently discover undeclared dependencies. Capture already provides
safetensors plus its argument manifest; quantized captures may additionally need their
inner-tensor payloads. Declare those too. These references do not make an audit directory
a standalone relocatable executable bundle.

`ir_audit=False` disables audits; `True` or `"exact"` retains exact snapshots;
`"compact"` retains exact hashes plus supported native views; `"both"` keeps both.
Invalid values refuse before lowering. CLI spelling is
`merlin-compile-kernel ... --ir-audit [exact|compact|both]` (bare flag: exact).

**Compact mode does not provide views for unsupported printers.** Supported xDSL stages
export large dense storage to content-addressed tensor files and record inspection-only
references with element types and shapes; see [model lowering](../guides/model_lowering.md).
Unsupported text-only/LLVM-IR records are hash-only in compact mode and explicitly marked
`unsupported-printer`. A wholly unsupported compact invocation warns visibly. Native
torch-MLIR workers inspect the same live operation immediately after parsing and before
emission using `get_asm(large_elements_limit=64, large_resource_limit=64)`, without
reparsing through another dialect runtime or changing pass boundaries. Both supported
views link to the observed exact native `get_asm()` serialization hash and the declared
sidecar index; exact native parents are also retained in `both` mode. Native formatting
may differ from original input text, so these hashes are not conflated.

Named-stage views are labeled inspection-only and use `.inspection.mlir.txt`; they are not executable
IR and never replace the executable input or output. Printer limits apply per attribute,
not to total output size. Ordinary executable build files are unaffected. No lossy elided
view is passed off as an externalized model. Portable sidecar copying, automatic compact
phase views and native-stage tensor export remain follow-up work. Completion rechecks
retained named-stage exact/inspection files and exported xDSL tensor payloads; this is
an end-of-invocation integrity check, not concurrent-writer exclusion.
Arbitrary MLIR attributes are not assumed to be representable by safetensors, and no
quantization/layout conversion occurs here.
