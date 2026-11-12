# AGENT.md — merlin/python/merlin/xdsl_dialects

## Purpose

merlin's own **core dialects** in xDSL. This is the default, fast Python plane for iterating on the `contract`, `schedule`, `interface`, `runtime`, and `dse` dialects before committing to stable MLIR/C++. Dialect namespaces are bare (no `m`/`merlin.` prefix).

## What belongs here

- xDSL dialect definitions, parsers/printers, verifiers.
- `_common.py` (shared enums/guards), `contract.py`, `schedule.py`, `interface.py`, `runtime.py`, `dse.py`.
- `_common.text` prints dense floating constants as raw-byte strings so xDSL and
  upstream MLIR retain bit patterns, including infinities, NaN payloads and signed zeros.
  Attributed standard tensor/Linalg yields use generic syntax because xDSL's
  shared custom yield printer places attributes before operands. Retain the
  actual operand, type and complete provenance when native readers reconstruct
  mixed-type named regions.
- `standard_tensor.py` binds standard `scf.forall`, `scf.forall.in_parallel` and
  `tensor.parallel_insert_slice` when upstream xDSL lacks them. These are standard
  operations, not target mapping or new Merlin semantics; native lowering owns complete legality.
- `standard_linalg.py` retains the standard named-matmul signed/unsigned cast
  property and enum missing from the partial Python reader. Native MLIR owns
  implicit mixed-type regions; this binding grants no new arithmetic policy or
  generalization. Keep explicit generic regions and named operation identities.
- `standard_scf.py` retains every execute-region attribute in both custom and
  generic syntax, including the native unit `no_inline` field. Its custom printer
  retains void terminators and provenance. Ordinary unmarked inlining delegates
  to xDSL; marked regions remain until an explicit lowering. The field grants no
  purity, borrowing or lifetime fact. Preserve entry/block/result verification.
- `standard_vector.py` binds the native optional scalar/string source and
  punctuation fields of `vector.print`. Retain its observable stdout effect;
  this binding grants no purity, buffer aliasing, physical replication or
  native runtime implementation.
- `fp8.py` binds the four TTIR builtin FP8 types and exact scalar/dense
  constants, with an optional `ml-dtypes` codec. The shared parser registry
  retains type identity with cooperating readers. Numeric operations, buffer
  ABIs and native/target legalization require their own executable contracts.
  Cooperating readers also remove false dense splats using complete element
  bytes, retaining mixed signed zeros while preserving existing NaN handling.
  Extend the complete float union captured by Linalg's abstract named-operation
  builder too; custom FP8 fills must construct their real scalar region.
- `standard_bufferization.py` checks standard tensor materialization by compatible
  shape/element types independently of memref layout, retaining upstream result
  and access-attribute rules. The binding grants no aliasing or placement proof.
  It also binds standard tensor views over compatible strided buffers.
- `standard_memref.py` binds standard reinterpret casts from ranked or unranked
  memrefs with checked element/space/layout and dynamic operand segments.
  This corrects a Python binding restriction without changing native semantics.
  It also binds standard generic atomic regions/yields with matching scalar
  element/index types, the required parent, and a pure modification body.
- `ir_inspection.py` — read-only generic printer views with large dense tensor attributes
  referenced by separate raw-byte audit payloads. Type, shape and hash descriptors retain
  their interpretation without converting numeric values. The no-sink printer still
  elides values. These views never replace executable input or change a function ABI.
- `lowering/` — the staged contract → schedule → interface → target → runtime lowering and its cross-op analyses, ending in a command-buffer dict for `merlin.runtime` (the Python engine).

## What does not belong here

- Stable production dialects (those would graduate to a future MLIR/C++ plane — not built; see
  `docs/design/compiler_plane.md`).
- Adapters to external xDSL tooling (implement in-package; see `docs/design/integrations.md`).
- Target dialects (generated into target repos by TargetGen).

## Interfaces

Prototypes the five core dialects (contract/schedule/interface/runtime/dse) in Python. Consumes/produces `merlin/schemas/` artifacts. `lowering/emit_command_buffer.py` emits dicts conforming to `command_buffer.schema.yaml`, executed by the `merlin.runtime` Python engine.

## Invariants

- These are rapid prototypes — expect churn, but keep `module.verify()` green.
- Stable dialects may later graduate to a future MLIR/C++ plane (see `docs/design/compiler_plane.md`); keep names aligned.
- `dse` is a real (minimal) dialect by explicit decision — it mirrors the `interface_candidate`/`dse_result`/`exploitability_report` schemas and never participates in lowering. `kernel`/`search` stay schemas-first.
- Local single-op checks live in `verify_`; cross-op checks (use-after-evict, placement legality, command-buffer consistency) live in `lowering/analyses.py`.

## Testing expectations

Per-dialect build/verify/invalid/round-trip tests plus the end-to-end lowering test under `merlin/python/tests/` (`test_xdsl_*.py`, `test_xdsl_lowering_e2e.py`).

## Notes for future agents

xDSL is the default way we handle MLIR-style IR until a specific pipeline is chosen. Copy the proven xDSL 0.65 idioms (field-annotation attribute params, `EnumAttribute` + `SpacedOpaqueSyntaxAttribute`, `func.ReturnOp`). See `docs/xdsl.md` and `docs/dialects.md`.
