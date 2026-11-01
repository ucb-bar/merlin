---
title: Compiler gate tutorial and supported profiles
kind: guide
status: current
owner: verification
last_verified: 2026-10-29
related: [lean_compiler_stack, getting_started, llvm_toolchain, merlin_lean_verification, compiler_verification]
code_refs:
  - src/merlin/verify/pass_validation.py
  - src/merlin/verify/compiler_transforms.py
  - src/merlin/verify/compiler_qdq.py
  - src/merlin/verify/compiler_macc.py
  - src/merlin/verify/compiler_coverage.py
  - src/merlin/verify/compiler_loop_source_profile.py
  - src/merlin/verify/compiler_buffer_source_profile.py
  - src/merlin/verify/compiler_configuration_source_profile.py
  - src/merlin/verify/compiler_ir_source_profile.py
  - src/merlin/compile/scheduling/configuration_ir.py
  - src/merlin/compile/scheduling/configuration_loops.py
  - src/merlin/compile/scheduling/configuration_c_consumer.py
  - src/merlin/llvmlower/lower.py
  - src/merlin/llvmlower/passes_xdsl.py
  - merlin/lean/lakefile.toml
---

# Compiler gate tutorial and supported profiles

This tutorial runs the integrated compiler gates, checks their results and
summarizes supported patterns. The [overview and status](lean_compiler_stack.md)
lists all ten implemented gates and their checked relations.

## Configure the tools

Start with the [Merlin setup](getting_started.md) and selected
[LLVM toolchain](llvm_toolchain.md). Use Python 3.12 or newer, the compiler's
authoritative `torch_mlir` environment, and the Lean 4.35.0 SDK used to build
the checker. The pinned exporter requirements are in
[lean-mlir-requirements.txt](../../build_tools/toolchains/lean-mlir-requirements.txt).

From the repository root, select the tools explicitly:

```sh
export MERLIN_COMPILER_PYTHON=/path/to/compiler-venv/bin/python
export MERLIN_LEAN_SDK_BIN=/path/to/lean-4.35.0/bin
export PATH="$MERLIN_LEAN_SDK_BIN:$PATH"
export MERLIN_VALIDATE_WORKERS=1
export LEAN_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

The checker library, checker executable and audit executable must all be current:

```sh
(cd merlin/lean && lake build --no-build MerlinVerify merlin-lean merlin-lean-audit)
```

If a build is needed, use the selected SDK and an explicit local resource budget.
For example, with two available CPUs:

```sh
(cd merlin/lean && LEAN_NUM_THREADS=2 taskset -c 0,1 nice -n 19 lake build MerlinVerify merlin-lean merlin-lean-audit)
```

Initial Mathlib setup needs its pinned dependencies and prebuilt cache; see the
project setup before starting a cold build. On Linux the worker also requires
`setpriv` for owned-process cleanup. Missing or stale tools fail validation.

## Enable the gates

For a normal compiler invocation, add:

```sh
merlin-compile ... --validate-passes all --lean-sdk-bin "$MERLIN_LEAN_SDK_BIN"
```

Or enable the same gate set for Python/compiler entry points:

```sh
export MERLIN_VALIDATE_PASSES=all
```

`all` includes MACC and Q/DQ. An explicit comma-separated list selects a subset.
Transformation options choose the rewrites; proof selection chooses the checks.
For example, `--macc-f32acc` requests the MACC transformation, while its gate is
already in `all`. Q/DQ expansion runs in normal preprocessing.

Python can also supply the original owner directly:

```python
from pathlib import Path
from merlin.verify.pass_validation import CheckerTools, PassValidation
from merlin.verify.speculative_gate import SpeculativeGate

tools = CheckerTools(Path("/path/to/lean-4.35.0/bin"), Path("/path/to/compiler-venv/bin/python"))
gate = SpeculativeGate(Path("out/build/example-gate"), job_timeout=900)
validation = PassValidation(gate, tools)  # all ten checks
```

Pass `validation` to `lower_model(..., pass_validation=validation)` or to the
selected scheduling consumer. Nested compilation joins its original owner;
a conflicting validation or a custom replacement for a canonical handler is
refused. Use fresh compiler-output and gate directories for each run.

## Q/DQ and MACC

A complete supported dequantization example uses runtime typed qparams:

```python
from merlin.llvmlower.lower import lower_model

source = """builtin.module {
  func.func @forward(%q: tensor<4xi8>, %scale: tensor<f32>,
                     %zero: tensor<i32>) -> tensor<4xf32> {
    %r = "quant_ext.dequantize_per_tensor"(%q, %scale, %zero)
      <{quant_min = -128 : i64, quant_max = 127 : i64}>
      : (tensor<4xi8>, tensor<f32>, tensor<i32>) -> tensor<4xf32>
    func.return %r : tensor<4xf32>
  }
}"""
result = lower_model(source, "out/build/qdq-example", targets=())
print(result.ll_path)
```

With `MERLIN_VALIDATE_PASSES=all`, the actual Q/DQ lowering is checked
automatically. The same gate covers complete textual preprocessing through
`lower_model(..., textual=True)`. No original capture or policy-approval flag
is needed to run the compiler lowering check.

For a supported bf16/f16 contraction, select its f32-accumulation transformation:

```python
from merlin.verify.compiler_transforms import Options

result = lower_model(
    source_with_half_contractions,
    "out/build/macc-example",
    targets=(),
    formal_transforms=Options(macc_f32acc=True),
)
```

The CLI equivalents are `--macc-f32acc --validate-passes all`, or
`MERLIN_BF16_F32ACC=1` with the same validation environment. The producer
extends operands to f32, accumulates in the original reduction order and
truncates once. Its actual supported before/after boundary must pass the
`bf16_f32acc_draft_contract` check. Existing producer calls made inside an
active validation scope are observed too.

`source_spec_approved=false` and `policy_reviewed=false` concern separate
source/numerical review. They do not disable either compiler gate.

### Original-source correspondence

To strengthen Q/DQ lowering with original-source correspondence, supply a fresh
qualified capture directory with `original.pt2`, `qdq-source.mlir`,
`qdq-frontend.json`, `qdq-producer.json` and `qdq-source-check.json`.
The capture driver is
[qualify_qdq_capture.py](../../build_tools/scripts/qualify_qdq_capture.py).
Select its pinned producer, Torch wheel, capture Python and dependencies using
`MERLIN_QDQ_PRODUCER`, `MERLIN_QDQ_TORCH_WHEEL`,
`MERLIN_QDQ_CAPTURE_PYTHON` and `MERLIN_QDQ_CAPTURE_DEPS`, then use
`--qdq-source-capture /path/to/fresh-capture` with validation.
`MERLIN_QDQ_SOURCE_CAPTURE` is the equivalent environment selection.
Saved `matched` fields alone do not qualify a source.

## Ordered loops

For source SCF loops, select unrolling, ordered tiling, or both:

```sh
merlin-compile ... --loop-unroll-factor 2 --loop-tile-size 3 --validate-passes all
```

The Python options are `Options(unroll_factor=2, tile_size=3)`.
`unroll_depth` and `tile_depth` select one-based nesting depths; the CLI flags
are `--loop-unroll-depth` and `--loop-tile-depth`. Tiling sees the module after
unrolling. An absent selected depth is unchanged and submits no proof job.

A loop from 3 to 31 with stride 2 has fourteen iterations. Factor two creates
seven ordered groups; tile size three groups those into segments of three,
three and one. Carries retain their original order. Word-loop composition and
generic-value singleton loops have different supported profiles below.

## Private buffer reuse

Select the existing allocator model and optional initializer tiling:

```sh
merlin-compile ... --buffer-reuse-allocator total_private_unobservable --loop-tile-size 2 --validate-passes all
```

The Python equivalent is
`Options(buffer_allocator="total_private_unobservable", tile_size=2)`.
Omit the tile to check normalization → reuse without segmentation. The compiler
captures three actual modules, or four with tiling, and checks the complete
chain. Identity normalization/reuse prefixes are supported.

These inputs already own their `memref.alloc`/`memref.dealloc` lifetimes.
The compiler uses its explicit-memory lowering route.
`Options(memory_source=True)` selects that route for an untransformed native
baseline. Tensor ownership inference, explicit tensor features and
vector/parallel schedules are not used on this route.

## Configuration elision and hoisting

These gates need the original selected ISA/effect facts. A caller label does
not supply instruction semantics. With `validation` from the setup example:

```python
from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection

selection = ConfigurationIRSelection(original_selected_contracts, "caller-effect-contract-v1")
result = selection.apply(original_module, pass_validation=validation)
consume(result.module)
```

The configuration stage of `compile_schedule` guards the actual straight-line
Kernel boundary and forwards that original owner through target emission.

For counted-loop hoisting:

```python
from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration

result = hoist_loop_configuration(
    kernel, adapter, selected_contracts=original_selected_contracts,
    pass_validation=validation,
)
consume(result.kernel)
```

For controlled signed-index64 C emission, select the consumer before the producer:

```python
from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer

consumer = ConfigurationCConsumer(index_bits=64)
result = consumer.hoist(
    kernel, adapter, selected_contracts=original_selected_contracts,
    pass_validation=validation,
)
c_source = result.emit_c(instruction_set, symbol="kernel")
```

The consumer revalidates selected contracts, endpoints and checked products
before emission. Recursive positive counted loops retain parent IV environments.
The indexed data proposal adapter is
`compiler_configuration_hoist_indexed.proposal_adapter`; configuration
dependencies and moved setters remain literal within that profile.

## Host graph fusion

Pass the selected host graph contract through
`lower_model(..., graph_optimizations=contract)` with validation enabled.
Its `fusion_groups` selects actual entry operation indices, including tensor
constructors. The gate captures the complete module before outlining and after
helper emission on either compiler route. It checks actual private calls,
live-in bindings and every public result, including producer fanout used by the
surrounding program. Earlier layout rewrites need their own checked boundaries.

## Read the result

A successful invocation writes final `coverage.json` after worker join.
For each boundary, inspect its selector, status, source/final hashes, profile
and owned job references.

| Record | Meaning |
| --- | --- |
| `verified` | A supported changing boundary has an accepted owned check |
| `unchanged` | Actual before/after bytes agree; no rewrite proof job or positive rewrite credit |
| `unverified` | A known original-source profile exclusion, with its reason and zero proof credit |
| Worker `not_established` | Lean did not establish the submitted supported obligation; compilation blocks |
| Worker `unavailable` | Tool, resource, process or evidence custody failure; compilation blocks |

An excluded source retains ordinary compilation; a failed target cannot be
reclassified as an original-source exclusion. A mixed invocation has the
individual claims shown in its journal.

Each job retains `request.json`, `verdict.json`, `worker.log` and proof
artifacts. Failure quarantines speculative `.ll`, `.o` and `.so` products.
A successful outer process alone is not evidence that every boundary changed
or earned proof credit.

## Supported profiles

All ten gates are enabled in `all`. The following domains describe their
current proof coverage.

| Family | Supported patterns | Bounds and separate obligations |
| --- | --- | --- |
| Tensor DCE/CSE | Declared index64 signatures and the Lean typed source-support rulebook, at complete actual module boundaries | Native primitive fidelity remains a model assumption; later canonicalization/LLVM CSE have their own scope |
| Ordinary memory DCE/CSE | Dead index constants in a source-owned memory epoch; authentic unchanged leading CSE | Constants and materialized dimensions must fit signed index64. Changed memory CSE has explicit unverified coverage |
| Word/index loops | Qualified inline/bulk/tail unrolling, carries, multiple changed definitions and selected nested depths; ordered tiling and supported unroll→tile composition | Literal nonnegative lower bounds, static fitting selected controls, positive strides, 2,048 weighted syntax records and a source-derived expansion budget |
| Generic-value loops | Exactly one top-level module `scf.for`, no carries, three immediate literal controls, one opaque body event, immediate captured return | UNROLL factor two with odd trip count at least three. TILE requires nonempty trips, positive stride and size 2–65,536. Generic unroll→tile composition and extra loops/events are outside this profile |
| Private buffers | Positive static rank-one/rank-two i8/i16/i32/i64 buffers, captured uniform initialization, nonescaping sequential lifetimes; untiled or tiled initializer chain | 256 KiB source, 5,000 operations, tile 2–65,536. Computed initializers, rank three, dynamic/floating storage, external declarations and callers reaching the changed kernel need further profiles. Successful unobservable allocator is an explicit premise |
| Q/DQ | Signed i8/f32 per-tensor Q/DQ with scalar i32/i64 zero points; supported per-channel DQ with i32 zero points | Per-channel i64 zero points and additional formats/ranges need additional profiles. Original-source correspondence and IEEE/framework fidelity are separate claims |
| MACC | Uniform bf16 or f16 static contractions, exact supported multiply/add body and projected maps, positive-zero initialization | Mixed formats, nonzero initialization and incompatible maps need broader compiler recognition. Checked f32-accumulation conformance does not approve a numerical policy |
| Configuration elision | Original selected instruction entries or straight-line Kernel calls under their ISA/effect contracts | 4,096 calls/256 fields and original selection bounds. Dynamic decoded operands, instruction control and floating instruction metadata need further profiles |
| Configuration hoist | Supported flat, sibling and recursive positive counted Kernel trees | 4,096 statements, 256 fields, depth 64. Ordinary extent cap one million; original C64 consumer supports positive signed64 extents symbolically. Indexed instruction/environment Models remain premises |
| Host graph fusion | One public entry, static unencoded byte-addressable integers, initialized inputs, unread destinations, pure identity-indexed pointwise stages and supported direct private calls; complete public outputs | One fused region of 2–128 stages. Floating arithmetic, reductions, broader indexing and multiple fused target regions need further profiles |

Tensor/QDQ/MACC original-source intake permits 4 MiB raw IR and 32,768
operations. Dense literals have separate aggregate limits of 64 MiB decoded
storage and 8,388,608 elements, charged before splat expansion. Runtime tensor
signatures do not consume the dense-literal budget.

Generic loop intake bounds raw IR/query/dense storage to 4 MiB, weighted
syntax/proof work to 65,536, blocks to 4,096 and nesting to 32. Word/private-memory
literal intake retains its separate storage bounds and refuses scalar NaN
identity loss. Resource failures block; only recognized source-profile exclusions
produce excluded coverage.

## Replay and standalone APIs

A saved success report is a locator, not live delivery authority. Replay
reconstructs the original inputs, builds fresh proof objects, audits roots and
imports, and binds the fresh completion to actual bytes and selected tools.
Copied tokens and edited success fields cannot authorize products.

| Boundary | Standalone interface |
| --- | --- |
| Tensor stages | `merlin.verify.lean.check` typed checks, source support and receipt-chain composition |
| Q/DQ | Canonical `compiler_qdq.check` worker; repeat compilation from the exact original inputs for a fresh whole-boundary check |
| MACC | `compiler_macc.check_boundary`, `deliver`, `replay` |
| Word loops | `compiler_loop_intake.prepare` with `loop_transform_gate.check/replay` |
| Private buffers | `buffer_reuse_intake.prepare_chain` with `buffer_reuse_gate.check/deliver/replay` |
| Configuration | `configuration_elision_gate`, `configuration_hoist_gate`; controlled replay through a fresh original `ConfigurationCConsumer` |
| Host fusion | `compiler_host_graph_fusion.check_boundary`, `deliver`, `replay` |

Use fresh destinations and the matching family's exact API/result type.
Historical producer→proof→consumer tools such as
`python -m merlin.verify.loop_transform_cli` retain their pinned producer
contract. They are developer interfaces, separate from the integrated
compiler-entry path.

## Troubleshooting and regression checks

| Symptom | Check |
| --- | --- |
| Missing/stale checker | Selected SDK/Python paths and the complete-library `lake build --no-build` check |
| Supported rewrite blocks | Actual captured before/after bytes, first worker diagnostic and bound source/tool hashes |
| No MACC job | Whether the producer ran, the source profile is supported and the boundary changed; policy review is not an enable switch |
| No Q/DQ job | Whether Q/DQ exists and changed; original-source qualification is an optional stronger check |
| Source exclusion | Exact profile/reason in final coverage; supported patterns are listed above |
| Replay refuses | Original input/tool/source bytes, fresh destination and the family's live-completion requirements |

Run the [compiler suites](lean_compiler_stack.md#coverage-and-qualification-status)
with `MERLIN_LEAN_REQUIRED=1` so missing proof tools fail instead of skipping.
Unset ambient `MERLIN_VALIDATE_PASSES` for these tests: they construct their own
validation owners and compare against ungated baselines. For example:

```sh
env -u MERLIN_VALIDATE_PASSES MERLIN_LEAN_REQUIRED=1 python -m pytest -q merlin/tests/ir/test_compiler_qdq.py
```

For native/reference testing of the semantics themselves, see
[Lean/native evaluation](lean_native_evaluation.md) and
[differential tests](lean_differential_tests.md).
