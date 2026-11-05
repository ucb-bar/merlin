# Shared connected C host execution

`merlin.compile.connected_host.emit_connected_host_c` consumes the checked
`ConnectedProgram` returned by `compile_connected_region`. Both an offline
compiler and an automatic author can call this API. See
[`connected_composition.py`](connected_composition.py) for the entrypoint.

Supply `emit_stage(invocation, symbol)` and explicit physical backing origins.
The callback returns `ConnectedCStage`: implementation source, borrowed-buffer
effects, input/output alignments and an identity for its window binding scheme.
It receives the actual checked dense product or exact pointwise images. Dense
products expose `emit_c`/`emit_llvm`; the callback supplies target instructions,
pointer binding and completion. Pointwise adapters must bind encoded addresses
to the provided logical views explicitly; an image and a pointer alone do not
establish relocation.

The generated C entrypoint accepts an array of `{base, bytes}` backings in the
receipt's order. All window sizes, pointer arithmetic, alignments, distinct
backing overlaps and descriptor aliases are checked before any stage executes.
Caller-provided memory must actually be accessible for its declared size.
The wrapper emits calls in source order and preserves every source live output,
including externally observed intermediates and fan-out consumers. Reusing
non-overlapping live ranges within one declared backing remains possible.

Status zero means every stage reported completion. A stage failure returns its
status and zero-based stage index, without launching successors; earlier stages
may already have written outputs. Argument refusal is `{-1, SIZE_MAX}`. Successful
completion is `{0, SIZE_MAX}`. Every return from a stage, including failure, must
obey its explicit noescape, no-release and synchronous effects. Status and effects
assertions do not replace independent output comparison or target qualification.

The receipt records the source/program identities, backing origins, window
offsets, byte orders, all live outputs and stage premises. There is no implicit
allocation, packing, byte-order conversion, on-chip residency or fusion. The
runtime is synchronous; overlapping device stages require a separate lifetime
and completion contract. Flat byte-addressable hosts with 8-bit bytes and
pointer widths at most 64 bits are supported.

The focused native tests in `merlin/tests/infra/test_connected_host.py` compile
the emitted wrapper with counted dense loops and exact pointwise images in an
independent neutral byte runtime. They compare all outputs, exercise relocation,
repeated invocations, mixed input/accumulator widths, refusal before launch and
failure propagation. An intentionally lying stage shows why a successful return
is insufficient. These checks establish neither accelerator execution nor speed.

For an isolated automatic run, select this module through the ordinary reviewed
compiler-library export before launch. Its dependency closure includes
`connected_runtime` and `llvmlower.borrowed_buffer_calls`; preserve existing
source hashes and review the new selection. Do not mutate a running author's
frozen library. The adapter remains authored in the fresh backend, and private
reference implementations and measurements remain excluded.

## Existing bounded host RNE in ordinary owned builds

An owned model backend may add `"host_bounded_rne": true` to its ordinary
`source_composition.json` request (alongside the required `schema`,
`shared_input_siblings`, and `prepack_weights` fields). The fixed host owner
forwards this to `OwnedHostModelCompiler.compile(host_bounded_rne=True)`.
It selects Merlin's existing late bounded clamp/RNE legalizer; it supplies
neither host LLVM nor numerical permissions from another experiment.

The original source must contain a recognized unconstrained binary32 bounded
integer rounding chain and the selected host lane must support scalar RV64GC.
The existing matcher, assembler, actual emitted-LLVM proof and complete-output
checks remain authoritative. No matching chain or unsupported CPU semantics
produce an explicit refusal. Defaults stay off; existing operator pass overrides
retain precedence. Applied feedback requires the pinned original preparation
report, rather than the request alone. This option does not select historical
interval-observer tables, source fallback replacements, or their FP effect
permissions, and does not imply a speedup.
