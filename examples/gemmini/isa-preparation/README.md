# Public command ISA data preparation

This operator tool produces a separate normalized facts document and an `isa`
declaration for the basic no-result RoCC command form. It imports only Merlin;
it neither reads nor creates a gemmini-mlir backend. Its generated package is a
**preparation check**, not an author seed. The fresh backend remains empty.

The original frozen facts are read-only. Run into a new output directory; existing
data with different bytes is refused. All input paths are explicit, so an artifact
evaluation can use relocated public hardware sources with the same byte hashes.

```sh
PYTHONPATH=src python examples/gemmini/isa-preparation/prepare.py \
  --facts "$PUBLIC_FACTS" --facts-sha256 "$PUBLIC_FACTS_SHA256" \
  --firrtl "$PUBLIC_FIRRTL" --isa-source "$PUBLIC_GEMMINI_ISA_SCALA" \
  --rocc-header "$PUBLIC_ROCC_HEADER" --rocc-header-sha256 "$ROCC_HEADER_SHA256" \
  --operand-protocol "$PUBLIC_OPERAND_PROTOCOL" \
  --operand-protocol-sha256 "$OPERAND_PROTOCOL_SHA256" \
  --out "$TASK_OUTPUT/public-isa-preparation"
```

Use Merlin's declared Python/xDSL environment and a host C compiler (`cc`). No
cross compilation, target execution, simulator, private kernel, or optimization
reference is used.

## Sources and products

- Original facts pin the selected FIRRTL and public Gemmini ISA Scala source.
  The script checks those hashes before extraction. The operand protocol must
  also bind the exact original facts and target.
- Existing `derive_operand_layouts` reads the actual Rocket instruction wire and
  eight destination slices from the original FIRRTL. Its receipt retains exact
  module/reference selections, source line numbers and full source hash. The
  32-bit instruction word is distinct from the two 64-bit register payloads.
- Existing `extract_funct_table` reads the declared Scala span (start comment and
  stop declaration taken from the original evidence). Names/selectors must agree
  with the selected observed decoder for every included command.
- The public RoCC header defines the no-result, two-source form. Its whole-file
  hash is independently supplied and retained. `checks/public_macro.c` evaluates
  that header's encoding macro using host C; it does not execute target commands.
- `normalized-facts.json` is a new, explicitly partial document with source pins,
  extraction receipts and limitations. Historical facts are never overwritten.
- `isa-declaration.json` references the new facts hash. `generation-check-only/`
  contains mechanically generated typed operations and codecs. The generated
  operations have two i64 SSA payload operands; an explicit allocation callback
  supplies their five-bit register indices. No payload builder is generated.
- `preparation-check.json` records 24 public-header comparisons: eight commands,
  each using zero, ordinary and upper-bound register indices. The generated
  operations are verified and encoded through the ordinary SSA allocation API;
  their memory-effect trait must remain unknown.

The normalized document can later be selected as a **new explicit input treatment**
with the matching declaration. Selection must retain the original hardware facts
as the hardware evidence source, bind the new document's provenance and update
only the ISA input identity. Do not replace the frozen hardware facts globally or
silently present this new document under their old hash. The script does not alter
any launch selection, software spec, or author workspace.

## Exact coverage and limits

Included selectors: CONFIG_CMD (0), LOAD2_CMD (1), LOAD_CMD (2), STORE_CMD (3),
COMPUTE_AND_FLIP_CMD (4), COMPUTE_AND_STAY_CMD (5), PRELOAD_CMD (6), LOAD3_CMD (14).
These are raw command forms, not qualified mathematical operations. Configuration
subtypes are payload data and are not split into ambiguous outer-word opcodes.

All other selectors are explicitly excluded from this initial subset. Selector
25 is source-declared but absent from the observed decoder table. Selector 126 is
observed but outside the selected unannotated declaration block (the public Scala
source separately names COUNTER_OP); neither is silently promoted to supported.
Counter result handling, loop commands and other forms require additional reviewed
operand/result declarations before inclusion.

Semantics and effects remain **unknown**, command acceptance remains **unproven**,
and full-ISA coverage is **false**. Register slices do not establish buffer
hazards, memory effects, command completion, legal reordering or numerical
behavior. The independent check establishes encoding agreement, not RTL execution
or performance. Graph lowering, payload packing, allocation, scheduling and safe
transformations remain compiler development work.
