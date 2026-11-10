# src/merlin/compiler/transforms

Own reusable transformations, their applicability analysis and materialization.
Preserve numerical ordering, effects, source lineage and explicit permissions.
Backend resources, instruction encodings and target-specific policies stay in OOT.

`host.py` composes native scalar transformations implemented by
`host_contraction.py`, `host_reduction.py`, `host_broadcast.py` and
`host_integer_math.py`. Their source literals execute only in the caller's
selected native context. Preserve the exact source bytes and concatenation order:
ordinary lowering runners and separately reviewed host-library releases consume
the same algorithms. Historical compile/LLVM-lowering imports are exact aliases;
new callers use these real implementation owners. Explicit numerical/effect
choices and independently selected host capabilities retain their obligations.

`pointwise.py` and `reduction.py` own portable tile geometry and byte transport
for explicit semantic requests. They preserve source affine maps, seed bytes,
padding and ordered boundaries without choosing arithmetic or target instructions.
The C transport consumers use these owners; semantic_compiler's historical modules
are exact aliases and published records retain their pickle names. Active native
snapshot receipts observe both aliases and actual implementations. Their packing
checks do not qualify a device kernel or authorize floating-point reassociation.

`contraction_k.py` preserves ordered reduction slices; `contraction_mn.py`
partitions independent output tiles; `contraction_mnk.py` composes those plans
and represents counted whole-program tiling. `panels.py` selects bounded streamed
M/N/K panels from explicit capacities and padding requirements. The caller must
qualify numerical continuation and padding; these owners do not infer identities
or target instructions. Legacy semantic_compiler modules are exact aliases;
published exceptions and records retain their pickle names. Native snapshots
observe actual implementations and aliases. Preserve reduction sequence, original
boundaries, tail/head padding and explicit accumulator-carry obligations.

`recomputation.py` owns bounded source recomputation. Its independent full-module
checker remains in `merlin.transforms.recomputation_check` during verification
ownership migration. Keep the checker and actual implementation sources pinned;
candidate construction is not execution, cost or hardware qualification.

Legacy module imports may alias the canonical module to preserve issued record,
function and monkeypatch identity. Historical SDK inventories remain immutable.

`multioutput.py` owns connected source fanout discovery and checked compilation
with two live outputs. Its independent semantic checker remains in
`merlin.compile.scheduling.multioutput_fusion_check`. Pin the actual algorithm,
compatibility alias and existing checker/source roster; the old module aliases
this owner exactly, including issued-record state. Preserve published record
pickle spellings. Missing composition permission retains individual lowering.

`row_normalization.py` owns exact row-island materialization and ordinary pass
orchestration. `row_normalization_source.py` reads closed source islands and
their affine row projections. Independent replay remains in
`merlin.transforms.row_normalization_check`; source admission is not its proof.
The historical module aliases the orchestration owner exactly. Its dispatch
bindings (including matcher and emitter) remain patchable there; pure matching
helpers now resolve globals in their defining source-analysis module and are
reexported as the identical functions. Both record classes retain historical
pickle names. Ordinary receipts observe both current owners, the old alias and
the existing checker/pipeline paths; these hashes do not attest arbitrary loaded
bytecode or provide live authority after the observation. Frozen SDK inventories
and historical full-source snapshots remain unchanged.
