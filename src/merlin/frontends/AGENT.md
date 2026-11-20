# AGENT.md — merlin/python/merlin/frontends

## Purpose

Frontends that ingest external IR into the Merlin pipeline. Today: linalg-on-tensors MLIR as produced by **model2MLIR** (`/path/to/model2MLIR`) — smolVLA and the other VLA workloads.

## What belongs here

- `linalg_mlir.py` — xDSL parsing of linalg-on-tensors artifacts + matmul inventory (shapes, dtypes, weights resolved via the safetensors manifest, `prov.*` provenance).
  Its standard memref and bufferization dialect registrations admit prepared
  fresh borrowed writer wrappers without an unregistered custom-syntax fallback.
  Standard scalar/string `vector.print` also retains complete registered
  fields and effects; its native software implementation is a separate gate.
  Standard materialization accepts compatible strided destinations with the same
  shape/element rules and writable/restrict/result checks as native MLIR.
  The standard SCF binding retains execute-region `no_inline` and all source
  attributes through custom parsing. Losing these fields can change later
  ownership preparation; never replace that preservation with a lifetime bypass.
- `linalg_named_regions.py` reuses the selected native parser/printer when a
  partial xDSL reader builds an invalid implicit named region. Retain the named
  operation, exact types/maps/attributes and source semantics; do not invent
  casts or generalize the payload. Explicit invalid source still fails native
  verification, and the restored complete module must pass Python verification.
  Valid generic regions and ordinary same-type operations remain unchanged.
- `facts.py` — lifting the inventory to contract-level reuse facts, driving the core pipeline with real model shapes, residency-variant DSE measurement.
- `compile_inputs.py` stages compiler-only Torch captures and explicitly derived tensor models.
  `compiler_constants.py` requires complete source-bound lifted constant bindings
  against original prepared placeholders, actual argument types and exact
  safetensors bytes. Retain the public parent ABI for derived models. Legacy
  unbound constants refuse; evaluator archives/digests never supply compiler
  values. The compiler-only projection grants byte custody, not numeric,
  effect, packing, ownership or target permission.
  `derived_compile_inputs.py` keeps parent FX and normalized TTIR correspondence separate,
  checks receipt-bound compiler members, and excludes evaluator values and digests from its staged view.
  Staging grants byte custody only, never source equivalence or numeric admission.
  `execution_model_correspondence.py` and `execution_call_contracts.py` own
  independent complete-SSA reconstruction of explicit v4 executed call binding.
  `execution_functional_state.py` checks the explicit functional mutation
  extension against original parent calls and actual retained readwrite buffers.
  Preserve complete ordinary/base slots, operand/result ties and entry state/
  user-result roles. Equal types cannot substitute a fresh output for state.
  Borrowed entry tensors remain readonly; state ports grant no physical public
  alias or automatic caller copyback. Never replace lifetime checks with tags.
  `execution_storage_views.py` checks v2's grouped mutable views against actual
  original TTIR pointer positions, offsets and complete cloned allocations.
  Each geometry field must reach logical i64 source formals; multiple arguments
  sharing one base must retain one readwrite state/result. Keep all original
  SSA and guards. Separate view copies cannot implement overlapping writes.
  `execution_call_views.py` additionally checks the explicit execution view
  profile, reconstructs dense value adapters and reuses original static
  source-coordinate/root proofs. Account for every view operand and check
  captured dimensions before reshape; compare complete emitted SSA and source
  locations. Missing profiles or forged coordinates refuse. No public physical
  alias/mutation or source-layout permission follows.
  Optional source-layout bindings reuse `t2m.captured-source-layout.v1` under
  v4: bind every source option to its distinct exact original scalar attribute,
  independently recheck the existing source-call contract, and require the
  serialized argument mapping to match. Missing profiles, unselected claims and
  changed source geometry refuse. Complete SSA/locations and lifetime checks
  still apply; source packing geometry grants no consumer hardware placement.
  `execution_runtime_views.py` independently reconstructs explicit v2 runtime
  dense views. Use actual kernel dimension SSA, guarded element products and
  standard collapse/expand. Same-rank dynamic types do not prove geometry.
  Empty domains must bypass product overflow; reject ambiguous output axes.
  Never guess coordinates from evaluator values or grant physical aliases.
  `execution_source_coordinates.py` independently reconstructs explicit v3
  runtime source-view paths and logical shape/stride/offset guards. Reopen
  actual transpose/slice/cast SSA and original allocation pointer bindings.
  Conditional empty slices require exact pure extract/empty branches, common
  dominating coordinates and an executable nonempty-condition check. Preserve
  complete model SSA and locations; records grant no physical alias or mutation.
  Rank-reducing slices follow standard reverse axis matching: only static unit
  sizes may vanish. Derive retained axes from mixed sizes and result types;
  never trust a serialized axis map. Preserve full offset contributions before
  projecting shape/strides, including scalar results and retained unit strides.
  Reopen original calls, retained located kernel bodies and every scalar/
  dimension/cast guard; compare the entire parent expansion and source locations.
  Located artifacts must also reproduce the retained unlocated implementation
  digest. No producer analysis, sample or result program supplies a fact.
  Derived receipt v2 now requires explicit execution-owned result selection and
  v4 binding. Stage the extra `kernels/<ordinal>/located.mlir` member only after
  ordinary source/memory custody checks and complete independent reconstruction.
  Public input axes come from actual types; original source-domain guards remain
  executable. No output-size program or static scratch estimate is substituted
  for the selected owned allocator. A source-only unqualified candidate may enter
  compilation; staging never reads its evaluator/qualification report and grants
  no execution verdict. Old receipt/default readers reject this selection.
  Correspondence grants no TTIR equivalence, numeric/effect, ownership or target permission.
  Explicit `execution_allocation_policy` on a derived v2 receipt selects
  `merlin.owned-result-limits.v2`. Independently validate complete finite positive
  byte/output/allocation-work limits and the supported index transport before
  signature v6 selection. Legacy v1 limits remain exact. These are operator
  resource bounds, not source shape facts or permission to skip allocation,
  ownership or lifetime guards; evaluators remain excluded from staging.
  `derived_live_arguments.py` independently checks the explicit
  `t2m.live-runtime-arguments.v1` profile against original typed parent calls.
  Exact scalar SSA and unencoded singleton tensors stay live; index casts need
  representable source constants/static dimensions. Captured and live scalars
  partition the source formals. Unknown runtime data cannot supply tensor
  extent/scratch facts, and earlier kernel inlining cannot invent original
  parent facts. Keep complete schema mappings, static public model types and
  evaluator exclusion; this grants no numeric/effect/alias or target proof.
  `t2m.checked-index-runtime-arguments.v1` additionally requires the original
  dominating unsigned index round trip and fixed-width comparison assertions.
  `source_index_ranges.py` reconstructs word enclosures without target index
  width, messages or tags; signed/unsigned extensions stay distinct. Check the
  exact adapter/range record and preserve executable guards. Range enclosures
  grant no concrete tensor extent, result-allocation or scratch fact. Older
  profiles refuse. `derived_symbolic_shapes.py` independently checks the explicit
  `t2m.symbolic-model-shapes.v1` profile in original call order. The local
  `model_shape_facts.py` reifies actual standard destination/result dimensions;
  no producer analysis import or evaluator value supplies a shape. Check kernel
  results and the complete inlined body against the original public contract.
  `merlin.compile-signature.v3` binds dimensions to public input axes, including
  actual parameter argument positions, without extra scalar arguments. Guarded
  scratch is the sum of actual checked source shape upper-bound costs. Missing
  profiles, false axes/body/range records and reordered dependencies refuse.
  General derived/data-dependent dimensions and physical/mutating views remain
  separate work; shape proofs grant no numeric, effect, alias or target permission.
  `t2m.symbolic-model-shapes.v2` separately admits typed derived size programs
  (`t2m.runtime-shapes.v3`, `logical-size-program.v1`). `shape_expression_ir`
  reopens actual source scalar/tensor operations and dominating assertions;
  `shape_expressions` preserves word widths, signedness and conditional values.
  Source-bound equality needs an actual bounded UNSAT proof when syntax differs.
  Missing/inconclusive solvers refuse. Signature v4 publishes that checked DAG;
  runtime allocation rechecks the complete body. Never import producer analyses
  or let a rehashed canonical program replace source proof. General runtime views,
  data-dependent allocation, resizing and effects retain their own obligations.
  `t2m.symbolic-model-shapes.v3` additionally selects typed public input words
  under runtime-shapes v4/program v2 and signature v5. `shape_input_words`
  independently proves actual singleton/native integer reads and dominating
  bounds, transferring only injective extensions. Check the exact live scalar
  and `integer_range` under `t2m.checked-word-runtime-arguments.v1`. The pure
  `shape_word_bindings` reader consumes values at invocation only, after the
  complete standard body has been rechecked. No input value supplies a compiler
  fact; false/rehashed bindings and older profiles refuse. General vector-data
  computations, runtime views/resizing, effects and targets stay separate work.
  `kernel_allocation_evidence.py` independently validates optional allocation
  source capsules: original TTIR, native-normalized/generic input, original
  argument bindings and source-only stage fingerprints. Copy only their exact
  compiler roster into standalone and derived-model staging. Reject evaluator
  fields in the stage projection, broken byte links and authority grants;
  retain legacy pointer packages without a capsule. Descriptor-origin bindings
  require a complete capsule. Custody grants no source equivalence or aliases.
  Capsule v2 additionally retains native generic original source, the printer
  identity, executable protected TTIR and the original typed contract graph.
  Check original/generic/protected links before copying; normalization must
  consume the protected source, and an unnormalized generic stage must consume
  that same protected source. Both v1 and v2 keep exact compiler rosters and
  exclude evaluator fields; v2 supplies no arithmetic or wrapper ABI permission.
  `kernel_barrier_evidence.py` checks the explicit `barrier_conversions` extension
  against actual typed original/protected source. Reconstruct single-program
  participation, address-space masks, phase scope/prefix/live results and
  original memory-event order, including order across intact phases. A retained
  location cannot replace a deleted store. No producer/evaluator import, numeric
  equivalence, alias, asynchronous/external effect or target permission follows.
  Barrier v3 additionally binds the explicit private volatile profile. Recheck
  every original volatile primitive and retained flag/type, including unused
  observations, and require the same bound memory profile. Phase order cannot
  replace a lost observation or grant cross-program publication.
  `kernel_cfg_evidence.py` independently validates optional original CFG custody:
  typed function input, native structured output, restored TTIR, complete original
  block/successor graph, fixed-pass stage and diagnostics. Bind all six compiler
  artifacts to the original source and trace in standalone and derived staging.
  Check the actual optimizer fingerprint and reject changed links or authority
  grants. This imports no producer/evaluator and grants no source equivalence,
  numeric, effect or target permission.
  `kernel_typed_storage.py` independently derives private byte-plane shapes and
  per-public-result guard costs for explicit typed storage transport. Public
  dtypes/shapes stay unchanged. Check the exact profile/function marker and
  maximum/vector costs; declarations grant no packing or source-equivalence proof.
  `kernel_literal_evidence.py` independently checks exact IEEE source-literal
  declarations for scalar/tensor/vector f16/BF16/f32/f64 constants. Bind storage
  words, dtype, shape, NaN count and primitive spelling to the complete source
  inventory, including protected allocation sources. Check tensor profiles before
  their memory-evidence early return. Custody grants no source equivalence or
  arithmetic semantics; executable integer constants/bitcasts remain ordinary
  compiler input, and evaluator data remains excluded.
  `kernel_compile_inputs.py` separately stages standalone functional tensor kernels,
  including explicit observable-effect receipts. ABI v6 permits void entries
  only under `effect_only: true`, with readonly inputs and no discarded writes.
  Preserve real result types in initially failing tensor cases. Outcome
  references, native observations and their individual digests stay outside
  compiler staging; source effects grant no implementation/target permission.
  validates declared buffer/result order, projects compiler provenance and records
  the entry-symbol adaptation to `forward` without inventing an FX capture.
  Its ABI v3 profile retains explicitly positioned by-value runtime scalars,
  labels scalar signature entries and excludes invocation values and tensor-only attributes.
  ABI v4 additionally retains representable signed integer range declarations;
  compiler-only staging grants no memory or guard-equivalence proof from those bounds.
  The explicit shared-functional-buffer payload retains the public functional ABI,
  checks byte-bound memory witness/continuation relations and copies their compiler evidence.
  Both standalone and derived staging exclude evaluator values from that view.
  Optional model2MLIR `tensor_words.npz` contains exact BF16 evaluator words.
  Exclude it and its digests with the ordinary inputs/goldens; byte-bound source
  correspondence supplies no permission to read those words during compilation.
  The explicit shared-guarded-buffer profile copies a byte-bound runtime
  admission report instead of a pure tensor witness. Standalone staging retains
  its declared guard-storage cost in the compile signature. It checks declared
  source/access/effect relations and grants no executable memory proof.
  Derived guarded kernels additionally bind per-call scratch costs to their
  typed results and model declaration. Runtime specialization reopens the
  original typed parent call by ordinal, checks scalar attributes, source
  operand/result mappings and representable bounded dimensions, then derives
  each call's concrete cost independently. Zero extents contribute zero; costs
  must fit i64. No evaluator archive/digest or producer-only shape claim supplies
  these facts. The evaluator budgets scratch separately from final outputs.
  Source model tensor views require `t2m.model-kernel-binding.v3` and explicit
  `t2m.captured-source-tensor-views.v1`. Independently recover static canonical
  transpose/positive rank-preserving slice coordinates from original parent
  SSA; compare every coordinate to captured typed integer kernel arguments and
  the exact executable allocation-root pointer binding. Account for every
  original schema tensor operand. Rehashed records, unrelated roots and profile
  downgrades refuse. Logical relationships grant no physical public alias,
  mutation or numeric permission; evaluator data remains excluded.
  Dense model value views require `t2m.model-kernel-binding.v2` (or v3 with
  source views) and its explicit
  `t2m.captured-dense-tensor-views.v1` profile. Reopen each original typed parent
  call. Optional semantic source-layout selections additionally require
  `t2m.captured-source-layout.v1`: bind all source geometry options to distinct
  original scalar attributes with exact types/values. Neither producer records
  nor a different source selection can substitute for the captured call.
  These checks grant custody, not GPU equivalence or consumer placement.
  For dense views, reopen the original typed parent
  call, including static kernels, and independently check the exact element
  type, unencoded static extents, equal element count and complete view records.
  Derive scratch from the kernel dimensions after captured scalar checks; zero
  and rank-zero parent values retain their real counts. This grants logical
  row-major reshaping only, with no physical stride, alias, mutation, ownership
  or numerical permission. Keep legacy v1 behavior and evaluator exclusion.
  V2 memory admission checks the sticky reader/writer protocol and derives
  metadata word count/cost from the maximum logical grid, retaining the explicit
  pre-CSE memory-effects stage. V1 evidence keeps its original scope and cost.
  The source-word buffer profile also retains exact word-stage input/output
  evidence and validates its explicit pass sequence and function access intent.
  Both retained legacy pipelines and the exact additional
  `math-expand-ops{ops=rsqrt}` computation pipeline are admitted; unfiltered or
  different math expansions remain refused. This grants stage custody only.
  V2 word evidence embeds isolated-event input/output, typed source bindings,
  arithmetic-only parent input and both actual native reports. Staging checks
  their byte chain, optimizer identity and declared inventory; producer-side
  event reconstruction and numerical equivalence remain separate obligations.
  Atomic view bindings use the complete source context and actual captured
  storage dtype. One source event may select integer and floating roots;
  a CAS template's logical pointer formal does not replace its storage capture.
  Primitive effect checks and producer reconstruction remain separate gates.
  Explicit byte-word atomic bindings separately retain actual i8 captures and
  integer template formals, one logical byte order, word size/alignment and
  byte/word address units. Independently match source primitives, complete byte
  guard tiles, actual storage and payload declarations; equal-width word-view
  permission cannot authorize this relation. Producer recipe reconstruction,
  executable ownership and numeric/source equivalence remain separate gates.
  V3 memory admission additionally retains the explicit private-invocation
  integer atomic contract, or the explicit v4 typed profile with bitwise f32/f64
  exchange and integer RMWs on equal-width floating storage, thirty-bit ownership fragments and source operation/
  scope/order/old-value inventory. Staging checks those declarations and costs;
  CAS additionally retains i16/i32/i64 comparison/old-value facts, its unmasked
  source access, and exact source versus memory opcode correspondence.
  Half/BF16 addition additionally binds format-specific nearest-even/quiet-NaN
  contracts to exact integer word widths, original storage and the full payload
  declaration. Equal bit widths do not permit swapping f16 and BF16 semantics.
  it grants no atomic linearizability, race/initialization or target proof.
  It shares the guarded public ABI without granting source or numeric proof.
  `kernel_shape_abi.py` validates ABI v5's named runtime dimensions and logical
  ceil-divided grids. Standalone compile-signature v2 retains symbolic tensor
  dimensions and typed scalar bindings. Staging checks declared types/relations,
  excludes example sizes and grants no memory, shape-guard or numeric proof.
  Named dimensions may occur on multiple axes with mixed static extents; dense
  stride/address and ownership proofs remain the producer's responsibility.
- `linalg_math_patterns.py` — source-only, closed unary f32 sine/cosine/exponential/roundeven
  scalar-region checks; each operation requires its own explicit declaration. No host admission,
  selected math implementation or numerical-equivalence claim follows. Scalar-literal power refuses.
- `linalg_mixed_patterns.py` preserves exact typed mixed source forms, including signed i32-to-f32
  conversion with its original scalar SSA edge and projected tensor maps. A source pattern does not
  supply a host declaration or certify conversion or execution numerics.
- `linalg_composite_math.py` — separate source-only, closed literal clamp, reciprocal,
  literal-base power, and tanh-GELU scalar regions. It records exact f32 literal bits
  and unresolved math-intrinsic obligations; no host, libm-supplier, or numerical grant.
- `linalg_patterns.py` has separate strict identity and opt-in static singleton-projected
  pointwise schemas. The latter reuses the same closed scalar-region proof, retains each
  positive-static input shape and exact affine map (including rank-zero scalar inputs),
  and grants no host placement, index-width bound, or numerical equivalence by itself.
  Projected i32 addition requires its explicit `integer_dtype: i32` source declaration;
  the closed proof rejects overflow flags and preserves the original integer width.
- `linalg_integer_reductions.py` — closed static i64 sum and masked prefix-sum source checks,
  with an explicit index-width premise; no host admission or linked-build verdict.
- `linalg_f32_maximum_patterns.py` — closed static one-axis f32 `linalg.reduce`
  with an exact negative-infinity seed and unflagged `arith.maximumf`. The
  selected index width is a caller premise; PyTorch reduction equivalence,
  host placement, and linked-build behavior remain separate obligations.
- `linalg_reduction_source_body.py` names the closed integer sum/prefix-sum/paired-minimum
  frontend/root-operation kinds shared by host admission and the independent linked
  witness. Its declaration validator grants no placement or numerical equivalence.
- `prepared_index_source_body.py` screens three exact tensor-index compute roles against
  a caller-reproved, receipt/trace-bound record. It validates parsed types, shapes,
  maps and ordinals but does not produce that record, grant placement, associate
  prepared nodes or establish original/compiled numerical equivalence.
- `bucketize_source.py` — a source-only, trace-bound proof of the closed f32
  bucketize counting reduction. The trace-bound count alone does not grant
  host placement, prove sorted boundaries, or certify generated code and
  numerical equivalence. Its separate opt-in static source-body recognizer
  requires a sorted non-NaN f32 literal
  and checks static index spans; the private issuer still owns right-flag
  ancestry and exact capture-to-build binding.
- `linalg_boolean_patterns.py` also owns a separate, closed dynamic rank-one
  i1-to-i64 unsigned cast source body. It proves input-derived output extent,
  identity maps and scalar extension, but not a runtime bound, host placement,
  external dynamic-output ABI or numerical behavior of a compiled program.

## What does not belong here

- Model capture/conversion (that is model2MLIR's job — don't vendor it).
- Lowering or dialect definitions (`merlin/python/merlin/xdsl_dialects/`).

## Interfaces

- Input: `workloads/<model>/<model>.mlir` + `<model>.safetensors.manifest.json` from model2MLIR.
- Output: `MatmulRecord`/`WeightReuseFact` inventories; `LoweringResult` via the existing pipeline; `dse` IR + `dse_result`-shaped dicts.

## Invariants

- Explicit `stage_compile_inputs(..., result_mode="execution-owned")` selects
  signature v6 after the original capture/kernel/model validations. Reconstruct
  actual public types and static dimensions; null dynamic result dimensions
  are observed after execution, never predicted from evaluator metadata. Keep
  original input restrictions, source locations and evaluator exclusion. Recheck
  v6 independently before compilation/invocation and reject older-profile
  reinterpretation. This grants no arithmetic, effect, alias or lifetime facts.

- Parse the **full** model artifact, not `sections/*.mlir`: the model2MLIR section splitter currently emits use-before-def SSA references (invalid SSACFG IR; e.g. `%2034` in `sections/smolvla.model.mlir`). Fix upstream before relying on sections.
- `PAREN_RESULTS` exists because xDSL 0.65's linalg parser rejects `} -> (T1, T2)`; remove it once xDSL accepts the parenthesized multi-result form.
- The integer pipeline executes a layer's **i8 deployment GEMM** with the real (M,K,N); the capture dtype is preserved as provenance, never silently claimed as executed.

## Testing expectations

`merlin/python/tests/test_frontend_linalg.py` — synthetic-text tests run everywhere; smolVLA-artifact and spike tests auto-skip when the artifact/toolchain are absent.

## Notes for future agents

smolVLA reuse structure: capture unit = one denoise step; `select_action` runs N steps per tick, so every weight is `reused_across_invocations` — the repeated-RHS pattern with real shapes.

`linalg_mixed_patterns.py` is opt-in typed routing source evidence. It proves closed
bf16/f64/mixed scalar SSA sequences, default tensor layouts and bounded static input
projections; it preserves finite literal bits and refuses init reads/fast-math. Historical
f32 schemas retain their meanings. It grants no framework numerical equivalence.

- `linalg_integer_seams.py` proves exact typed clamp/truncation and low-bit XOR source bodies; it grants no numerical qualification.

- `linalg_quantization_seams.py` proves exact scalar reciprocal and projected FP32 roundeven/clamp/int8-cast source bodies. Native numerical and selected rounding obligations remain separate.

- `connected_region.py` extracts an exact single-block source window with literal/init closure, nested captures and every live output. `verify_window` reconstructs from pinned original bytes before returning operation pairs. Re-run source/trace/index proofs on the projection; do not copy ordinal-bound host admissions. Runtime inputs remain explicit, and no numerical or target qualification follows.

- `bounded_control_source_body.py` admits only exact scalar count/index/guard roles from a caller-reproved whole-source interval context. `connected_index_source_body.py` preserves the closed prepared-index recognizers while requiring a separately reconstructed projection and exact captured typed-root join. Neither supplies linked host qualification, compiled bounds, or numerical credit.
