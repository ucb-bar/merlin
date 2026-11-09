---
title: Explicit provider compilation and native profiling
kind: guide
status: current
owner: core
last_verified: 2026-11-08
related: [getting_started, architecture, profile_evidence]
code_refs:
  - src/merlin/llvmlower/cli.py
  - src/merlin/llvmlower/provider_workflow.py
  - src/merlin/compile/provider_execution.py
  - src/merlin/runtime/backends/device_simulator.py
  - src/merlin/runtime/profile_evidence.py
---

# Explicit provider compilation and native profiling

`merlin lower` can compile an existing typed integer MLIR file through a selected
support provider, build a native simulator image, check every output through an
explicit independent checker, and retain observed region profiles. This is an
opt-in file workflow. Normal `merlin lower`, model capture, `merlin-compile`, and
whole-model routing defaults remain unchanged.

Install Merlin's compiler dependencies as described in [Getting started](getting_started.md).
Select an executable support provider with `MERLIN_TARGET_PATH`. The provider must
implement `im2col_stream_api()` and `copy_loops_api()`; the published RVV provider
at `f14004889757a39379e55280ed806a99aa9ed8d7` supplies both. Source normalization
and exact-divisor scheduling additionally require the provider's
`integer_file_api()`. A provider lacking a selected API refuses instead of using
another implementation. Configure the native
LLVM/MLIR install with `MERLIN_MLIR_INSTALL` and the interpreter containing
`torch_mlir` with `MERLIN_COMPILER_PYTHON`. Preserve the interpreter's virtualenv
path even when it is a symlink. No target checkout is discovered by unit name.

## Compile supplied IR

Create an explicit provider configuration:

```json
{
  "schema": "merlin.provider_workflow.v1",
  "target": "rvv",
  "vlen_bits": 256,
  "elen_bits": 64,
  "operand_bits": 8,
  "accumulator_bits": 32,
  "operand_lmul": 1,
  "workspace_bytes": 65536,
  "normalize_windows": true,
  "canonicalize_contractions": false,
  "exact_register_blocks": false,
  "stream_im2col": true,
  "copy_loops": true,
  "timeout_seconds": 120
}
```

VLEN, ELEN, SEW, LMUL and the software workspace budget have separate meanings.
The example values are caller choices. The provider derives register geometry
from them; it does not infer DLEN, cache capacity or physical utilization. Supply
`cache_budget_bytes` only when a caller owns that budget. Streaming and copy
lowering default to false, and require explicit selection in each config.

```bash
merlin lower source.mlir --out "$MERLIN_OUT_ROOT/build/provider-001" \
  --provider-config provider.json --target riscv
```

The preparation and lowering order is:

1. Run ordinary shared typed preprocessing and retain `model.upstream.mlir`.
2. If selected, call the provider's `integer_file_api()` using Merlin's existing
   contraction normal form to normalize a
   closed signed-I8, zero-seeded I32 window MAC. Retain `model.prepared.mlir`.
   Its public output order remains unchanged. Overflow flags, unsigned windows,
   nonzero seeds and other unsupported source bodies refuse normalization.
   `canonicalize_contractions` additionally normalizes closed transposed matrix
   contractions. `exact_register_blocks` selects provider-owned power-of-two
   exact M/N divisors, bounded by the supplied architectural geometry. This
   explicit conservative schedule may use one-element tiles for prime extents;
   it is not an optimal-kernel claim. Streaming and exact-divisor scheduling
   are separate alternatives in this initial workflow.
3. Call the provider's public streaming API. It owns matching, bounded panels,
   tails, alias/lifetime rules and the vector schedule. Declined cases retain
   their source and a counted refusal. Normalizing a direct convolution does
   not itself prove that the provider accepted streaming.
4. Call the provider's public copy-lowering API with the complete native
   pipeline. Unsupported copies retain upstream lowering and their refusal.
5. Use ordinary `mlir-translate` and object codegen with the selected VLEN.

`provider-workflow.json` records actual selections, counts, emitted IR/object
hashes, source/tool bindings and commands. The command's JSON includes those
counts and receipt paths. Reaching a compiled object grants no numerical or
performance claim. Mixing loaded Merlin code from different checkouts refuses.

This closed native path currently accepts integer IR. Floating policy, shared
runner features, textual preparation and IR-audit flags continue to use the
ordinary lowering path; combining them with `--provider-config` refuses instead
of silently dropping their callbacks. This path does not quantize a captured
FP32 model or enable selective mixed precision automatically.

## Select a matrix provider

An optional `matrix` object has `support_target`, `unit`, `config` and optional
`kernel`. Its support target must match the selected provider chain. The shared
workflow delegates geometry, selection, source rewriting and object construction
to that provider's public matrix plugin. Unselected contractions remain CPU
work, and a route count of zero is reported explicitly.

With `stream_im2col=true`, the matrix provider must expose
`streaming_selector(unit=..., config=..., vlen_bits=..., workspace_bytes=...)`.
An optional `streaming_options` object forwards `panel_rows`, `prepacked_a` and
`admission`; the provider owns their meaning and validates them. Do not also
select shared `normalize_windows`: a matrix provider owns its source orientation
and ABI. Generic RVV panel/copy algorithms remain in the RVV provider, while
device ABI and placement remain in the selected matrix provider.

## Build, execute and check a native image

An execution plan is a JSON object with schema
`merlin.provider_execution.v1`. Every input file uses
`{"path":"...", "sha256":"<64 lowercase hex digits>"}`. Relative paths resolve
against the plan directory. Required fields are:

| Field | Meaning |
|---|---|
| `compiler` | Exact compiler/link driver executable |
| `sources` | Complete C/assembly software harness source roster |
| `objects` | Optional prebuilt software support objects |
| `linker_script` | Explicit image memory/layout contract |
| `compile_flags`, `link_flags` | Literal argv lists, with no shell evaluation |
| `entry_abi` | `symbol` and ordered `argument_ranks` for the default-C, void-return flattened memref entry |
| `simulator` | Exact JSON native simulator descriptor |
| `isa`, `harts`, `active_hart_ids` | Explicit global ISA, physical hart count and compute worker roster |
| `strict_postgate` | Bound `python` and `script` files for the complete-output checker |
| `input_sha256` | Additional source/header/golden/manifest file bindings |
| `timeout_seconds`, `postgate_timeout_seconds` | Explicit finite process bounds |

The compiled LLVM entry must match the bound harness's flattened pointer/index
signature before any native execution. The harness owns startup, traps, console,
allocation, completion and all input/output storage. Its runtime, header and
library closure must be supplied explicitly; compilation does not infer an
allocator or memory capacity.

The simulator descriptor owns its executable identity, launch arguments, memory
regions, optional per-hart ISA, worker restrictions and qualification. The runner
preserves those fields and refuses incompatible physical/worker placement. There
is no stock-simulator fallback. It bounds and cleans up only its own child
process group, retaining raw stdout/stderr on failure. This is a native simulator
workflow; it performs no FPGA lifecycle or hardware allocation.

```bash
merlin lower source.mlir --out "$MERLIN_OUT_ROOT/build/native-001" \
  --provider-config provider.json --target riscv --execution-plan execution.json
```

The checker receives `--console FILE --output FRESH_DIR --counter-semantics NAME`.
It must exit zero and write `report.json` with `pass: true`. It must independently
verify the intended complete outputs, input/guard protocol and terminal markers;
a guest exit alone is insufficient. Additional dependencies of that checker
belong in `input_sha256`. The receipt says `passed_supplied_gate`, preserving the
gate's actual method and limitations rather than granting certification.

## Retain observed profiles

Select `profile` in the execution plan, for example:

```json
{"expected_regions": 1, "record_path": ["regions"]}
```

The postgate's region rows must contain checked counter endpoints. An optional
bound `labels` sidecar follows the [external profile evidence](profile_evidence.md)
contract and binds the exact final ELF before assigning source labels. No
function self-time or unknown attribution is inferred from a filename.

The workflow writes `execution/profile/report.json`, `regions.csv`, and
`trace.json` in Chrome/Perfetto format. These are actual observed intervals.
Functional instruction counts stay distinct from experimental modeled cycles;
neither domain establishes physical busy time, traffic, bandwidth or calibrated
timing accuracy. No timeline is reconstructed from aggregate means. Existing
timing descriptors can be selected unchanged in a separate execution plan;
changing the simulator or its timing inputs creates a separate cohort.

All destinations are fresh. Raw failures, postgate refusals, source snapshots
and earlier receipts remain available for review. A new result never overwrites
an existing output directory.
