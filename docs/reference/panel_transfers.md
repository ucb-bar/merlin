---
title: Affine panel packing and transfer aggregation
kind: reference
status: current
owner: core
last_verified: 2026-10-20
related: [architecture, shared_pass_phase2]
code_refs: [src/merlin/compile/scheduling/panel_transfers.py, merlin/tests/infra/test_panel_transfers.py]
---

# Affine panel packing and transfer aggregation

`merlin.compile.scheduling.panel_transfers` derives packed panel addresses from an
immutable two-dimensional source view and explicit storage and transfer contracts.
Adjacent primitive tiles can share a wider packed row, so transfer width can exceed
compute width. The caller chooses panel dimensions; the pass checks those choices
against supplied capabilities without discovering a target or choosing instructions.

Existing `block_schedule.gather_runs` groups equally spaced gathered source rows at
one column. `window_transfer.Window2D` constructs an affine rectangle predicate for
runtime convolution coordinates. This family addresses the separate column-layout
problem: represent adjacent tiles in one packed panel and combine compatible actual
source/destination row segments into rectangles. It does not replace either mechanism.

## Inputs and guarantees

`AffineMatrixSource` identifies a source backing, a positive logical shape, an element
width, an allocation bound, and the first byte and row/column byte strides. Transposed
and broadcast views are supported. Noncontiguous source columns become separate scalar
byte copies; a wide source copy requires proven contiguous columns. Negative strides
and unknown bounds refuse.

`PackedPanelLayout` selects a primitive tile shape, tile counts per panel and row
alignment. Panels follow matrix row/column order. Each panel contains consecutive
packed rows, including exact byte-zero initialization of tail rows, tail columns and
alignment gaps. It requires an assigned, aligned, invocation-private `StorageWindow`
with known capacity, after reservations. Source/destination backing overlap refuses.

`PanelTransferLimits` declares maximum rectangle rows and row bytes, address/length
alignment, source/destination stride limits and exact byte-zero support. Missing stride
support keeps single-row copies. Missing zero-fill support permits only padding-free
layouts. Unaligned tails refuse rather than reading neighbouring source bytes. These
limits must come from the selected instruction/software semantics and hardware facts;
storage size or a preferred compute tile cannot supply transfer capabilities.

`resolve_panel_transfer_limits(declaration, facts)` connects those values to the
selected software and RTL inputs. Its exact versioned declaration has this form:

```json
{
  "schema": "merlin.panel_transfer_limits.v1",
  "limits": {
    "max_rows": {"fact": ["copy", "maximum_rows"]},
    "max_row_bytes": {"fact": ["copy", "maximum_row_bytes"]},
    "address_alignment": {"fact": ["copy", "address_alignment_bytes"]},
    "row_bytes_multiple": {"fact": ["copy", "length_alignment_bytes"]},
    "max_source_stride_bytes": {"fact": ["copy", "maximum_source_stride_bytes"]},
    "max_destination_stride_bytes": {"fact": ["copy", "maximum_destination_stride_bytes"]}
  },
  "semantics": {
    "copy": "exact_bytes",
    "completion": "synchronous",
    "zero_fill": "exact_zero_bytes"
  }
}
```

The paths are software-selected structural paths into the supplied facts, using
the existing ISA resolver. Other fact layouts need different paths, including
integer list indices where appropriate. All six numeric fields require resolved
integer facts; software literals, missing paths, unknown bounds and inferred
defaults refuse. Zero stride bounds explicitly disable multi-row transfers.
`zero_fill` must be `exact_zero_bytes` or `unsupported`; unknown, converting and
asynchronous transfer semantics are outside this family. The caller retains
custody of its selected declaration and facts. The resolver supplies no execution
qualification, instruction implementation, ownership or permission authority.

`plan_packed_panels` returns actual logical byte transfers. Its independent
`check_panel_transfers` checks every projected rectangle row against the original
affine source map and the packed destination coordinates, then proves complete
destination coverage exactly once. Budgets bound row segments before enumeration;
they do not authorize an unchecked fallback. Its host executor reads and writes the
actual supplied byte addresses, without recovering saved planner requests.

## Consumer boundary

A newly reviewed Phase 2 library may expose this family. The generated compiler
consumer still has to bind its source view to the current typed source, realize
packing or transfers, and project actual emitted accesses back into the checker.
Library membership and these logical checks confer no import grant, target ownership,
instruction qualification, completion guarantee, or hardware performance result.
An existing frozen Phase 1 package remains unchanged; target compiler edits occur
only through the normal Phase 2 authoring path.

The consumer must establish that its compute addressing can read the packed tile
subviews, using `tile_first_byte` and the declared packed row stride. A banked transfer
with a different physical mapping cannot be represented as a linear rectangle merely
because its nominal width matches. It needs its own actual-address projection or
another layout. The pass does not infer row/bank mappings from capacity.

Padding contains zero bytes and does not enlarge the original reduction or change
quantization semantics. A consumer must preserve original valid extents and numeric
boundaries, or independently establish its arithmetic interpretation of padding.
Transfers are synchronous logical operations; this family supplies no overlap,
barrier removal, DMA latency model, runtime cache lifetime or constant-packing policy.

Phase 2 can tune panel tile counts, row alignment and compatible pass composition.
Transfer/storage limits remain capabilities and cannot be tuned beyond their facts.
The tests compare actual destination bytes against an independent original-data
oracle across several geometries, tails, padding, transpose and broadcast views,
including deliberate source/stride/destination/coverage corruption. Command-count
reductions in these tests are algorithm evidence, not measured accelerator cycles.
