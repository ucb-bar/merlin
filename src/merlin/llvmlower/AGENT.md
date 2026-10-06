# AGENT.md — merlin/python/merlin/llvmlower

## Purpose

Whole-model lowering: linalg-on-tensors MLIR (model2MLIR artifacts) → upstream MLIR pipeline → LLVM IR → x86 (verification) / rv64gcv (deployment) objects. This is the llvm-project plane for running entire models (smolVLA) on RVV, complementing the per-kernel `runtime/backends` path.

## What belongs here

- `cli.py` exposes the existing file-lowering API through `merlin lower`, without
  capture, optional research workflows or deployment. It requires fresh output,
  defaults to LLVM IR only, and forwards audit/sidecar options. `LowerResult.audit_index`
  identifies the invocation's exact audit index; callers must not guess the latest child.
- `passes_xdsl.py` — Merlin-authored rewrites: `quant_ext.dequantize_per_channel` → `linalg.generic`; `llvm.emit_c_interface`; (future) `scf.parallel` → `merlin_parallel_for`.
  Structural preprocessing accepts the invocation's shared IR audit and records parsed
  input plus each completed rewrite. Inspection must preserve pass order, statistics and
  executable serialization; failures keep only the completed prefix. The textual repair
  path is not re-parsed merely to create diagnostic views.
- `torchao_affine.py` — decompose the two torchao activation-quant ops a dynamic-activation capture leaves as opaque `func.call`s to body-less externs (`choose_qparams_affine`, `quantize_affine`) into linalg. Bit-exact against torchao's own implementation (`merlin/tests/ir/test_torchao_affine.py`); the block layout comes from the call's types and the quant range/eps from the scheme the bundle records in `prov.quantization`, and anything underivable raises rather than defaulting. Runs first in BOTH `dispatch_runtime.run_model` and `zephyr_model._prepare_model_mlir`.
- `pipeline.py` — upstream pass pipeline + `translate_module_to_llvmir` (in the model2MLIR venv).
- `op_profile_structural.py` reads typed top-level function operations for the
  existing optional profiler. Normal model backends cover effect-only calls and
  stores as well as result-producing operations, retain typed source identities,
  and accept generic function printing. Nested regions belong to their enclosing
  interval; unsupported multi-block entry control flow and existing marker symbols
  refuse. Input modules are cloned. Marker-induced optimization/hoisting changes
  require a paired uninstrumented timing comparison before using costs.
- Textual preprocessing attaches the C interface to a generic-format public
  function definition through typed IR. Private and external profiler/provider
  callbacks retain their original ABI. Generic printing must not silently lose
  `_mlir_ciface_forward` at the upstream boundary.
- `lowering_recipe.py` records ordinary lowering commands, resolved pipelines/gates,
  selected features, exact prepared input/runner/schedule identities and the redacted
  project environment. Its success hash binds the returned LLVM text, before later
  host transforms/codegen/link. This observation is not a hermetic toolchain lock or
  cache identity; secret and oversized environment values remain redacted.
- `late_quant_rne.py` owns reusable bounded binary32 round/clamp recognition on
  emitted LLVM, an explicit typed SSA tokenizer, portable native emission and
  explicitly selected CPU ISA legalization. The empty host policy leaves source
  bytes unchanged. Its verified callback runs before normal object identity;
  accelerator instructions, resource facts, ABI glue and legacy relink/audit
  tools remain in the selected OOT provider.
- `scalar_pointwise_unroll.py` owns a default-off upstream LLVM partial-unroll
  hint for structurally recognized innermost scalar f32 FMA/division loops. It
  changes loop metadata only; arithmetic, memory dependencies, existing strict
  scopes and source numerical contracts remain authoritative.
- `scalar_squared_sum.py` provides explicit typed ordered tensor squared-sum
  accumulators in scalar SSA. Static permutation input maps, complete output
  projections, the original multiply/add operand order and initial tensor values
  are retained; empty reductions return their source initialization. Strict FP,
  fastmath and unproved bodies refuse. Tensor semantics and upstream bufferization
  govern aliasing and lifetimes. No target or workload policy is implied, and
  complete emitted allocation/copy/store costs determine profitability.
- `scalar_contraction.py` also offers an explicit two-row/four-column schedule.
  Exact projected input maps prove immutable coordinate reuse across rows and
  columns; source multiply/add order, seeds, output-coordinate permutations and
  increasing reduction order remain unchanged. Partial tiles, strict FP and
  unsupported bodies refuse. Eight scalar accumulators are a portable scheduling
  choice, with upstream tensor alias authority retained; target ISA and workload
  identity never select it. Default source lowering is unchanged. Unconstrained
  source arithmetic retains LLVM's allowed NaN payload propagation choices;
  arithmetic payload identity is not an added contract. Nonarithmetic empty-K
  initialization words remain unchanged, and strict FP contexts refuse.
- `bufferized_result_identity.py` offers explicit upstream canonicalization
  before public result conversion. It exposes unchanged memref identities
  carried by loops so upstream can forward complete allocations into output
  arguments. Tensor alias analysis and ownership remain authoritative; no
  physical no-alias or floating permissions are added. Empty feature selection
  preserves the original pipeline, and complete emitted allocation/copy costs
  determine profitability.
- `bounded_rne_basic_block.py` is an explicit CPU legalization for two through
  four adjacent, independently proved bounded RNE results. Raw inputs must be
  defined earlier in the same block, and no statement, label, memory operation
  or unknown call is crossed. It preserves other uses of original source chains,
  refuses strict/constrained FP and verifies target plus portable LLVM before
  normal object identity. This mechanism does not select a profitable width;
  runtime measurements govern promotion under an explicit host CPU policy.
- `scalar_pointwise_packet.py` offers explicit independent scalar lane packets
  before bufferization for pure static all-parallel tensor bodies with existing
  FMA chains and division. It preserves source arithmetic per lane, projected
  input maps, bounded tails and tensor alias semantics; output-init reads,
  unknown scalar operations, constrained FP and fastmath bodies refuse.
  Bufferization copies, allocation and register pressure must be measured.
  Its separate broadcast-axis choice requires a used nonscalar input omitting
  that axis, shares only identical tensor extracts and retains exact axis tails.
  All source dimensions remain parallel; loop interchange grants no new alias
  or floating permissions. Profitability needs complete generated-code timing.
  A separate default-off multiplication selector accepts at least three source
  f32 multiplies, scalar integer quantization and static zero coordinates on unit
  input axes. FMA, division, precision changes and strict scopes refuse. It can
  compose with the disjoint FMA/division selector; both preserve original scalar
  operations and upstream tensor ownership. A local win alone does not establish
  whole-model profitability or an automatic schedule policy.
  `packet_scalar_pointwise_two_multiplications_4` is an independent default-off
  family for f32 tensor results with exactly two source multiplications,
  constants, optional integer casts and additions. It preserves every rounded
  source operation and refuses other arithmetic, output-init reads, strict FP
  and fastmath. It does not broaden the existing at-least-three selector; the
  two multiplication families and FMA/division family are structurally disjoint.
- `constant_fma_packet.py` analyzes consecutive, independent typed binary32
  FMA calls with exact finite constant words under an explicit ordinary,
  nontrapping, exception-flags-unobserved source policy. It retains the complete
  LLVM source/context witness, source operand order and original result SSA
  identities. The default leaves source bytes unchanged; an explicitly supplied
  OOT provider owns CPU ISA legality, register constraints and emission. Packet
  width and constant lifetime choices require measured complete-body costs.
- `broadcast_math_hoist.py` explicitly materializes pure source reciprocal-square-root
  chains on their exact smaller broadcast domain after elementwise fusion and
  before bufferization. Typed maps and SSA establish invariance; every cast,
  intermediate precision and live use is preserved. Empty/dynamic domains,
  unknown effects and strict FP refuse. Emitted call names never grant purity;
  additional allocation and traffic remain part of qualification.
- `tensor_preparation_identity.py` provides read-only preparation opportunity
  analysis using immutable tensor SSA roots and exact same-rank static slice
  coordinate composition. Equal shapes do not establish identity; encoded,
  dynamic and rank-changing views refuse. It retains typed source/context and
  consumer witnesses with sequential dominance checks. Explicit format hashes
  keep distinct preparations separate. No packing, call-purity, physical alias,
  reuse lifetime, runtime cache or cost permission follows from this census.
- `llvm_loop_outline.py` owns explicit post-bufferization LLVM loop extraction.
  The default-off feature derives new function symbols with LLVM tools and keeps
  only extracted helpers out of line. It adds no floating reassociation permission;
  compiler time and helper call overhead must be measured for each candidate.
  The separate default-off `outline_llvm_loops_merge_identical` policy follows
  extraction with LLVM `mergefunc`, checking that every original function symbol
  remains defined. Only LLVM's exact function comparison grants merging; source
  arithmetic, observable stores and public function-address rules remain intact.
  Smaller code does not establish a cycle improvement.
- `immutable_llvm_base.py` offers explicit immutable global-address binding
  through an unchanged public wrapper to a hidden out-of-line implementation.
  Typed pointer-use closure permits only GEPs and nonpointer loads from an
  ordinary constant global. The original operations, order, attributes, pointer
  identities and public signature remain intact; no FP, memory, effect or alias
  permission is added. The build must seal the extra hidden borrowed-pointer ABI
  and verify the complete resulting module. This default-off scheduling choice
  does not establish a register or timing benefit: actual rematerialization,
  call frame and complete-body costs require qualification.
  Explicit LLVM frame/caller/stack/coroutine observations, special function
  prologues and escaping block addresses refuse. Opaque calls and assembly keep
  their existing provider obligations for helper placement; callee names grant
  no purity or frame-independence fact.
- `llvm_loop_metadata.py` retains an explicitly chosen ordinary CPU loop through
  upstream translation/optimization. Callers identify the actual latch; existing
  annotations refuse pending explicit composition. It grants no arithmetic,
  floating, dependence or alias permissions. Providers select profitability and
  keep accelerator command encodings in OOT.
- `ordered_fma_matmul.py` provides opt-in independent-output scheduling for
  source-proven zero-seeded increasing-K f32 FMA contractions. It supports f32
  and exactly widened BF16 inputs, matching static batch dimensions, explicit
  RHS transposition, and separate bounded tails. Output tiling never grants K
  reassociation; callers establish the finite-intermediate and RNE contracts.
  Explicit BF16 operand widening may select neither operand, either one or both.
  Each selected operand is copied exactly once to f32 before the contraction;
  callers must account for its allocation, conversion, lifetime and traffic.
  The default performs no packing and no automatic policy selection.
- `ordered_fma_rewrite.py` explicitly schedules canonical tensor FMA generics
  using typed affine maps, iterator order, scalar wiring and positive-zero seed
  proofs. It supports matching static batch dimensions and explicit physical RHS
  transposition. Unsupported indexing, initialization or arithmetic is retained.
  Source identity attributes survive rewriting but do not select operations.
- `bounded_rne_lanes.py` emits explicit portable or CPU ISA helpers for bounded
  binary32 clamp/RNE packets. Callers prove original bounds, numeric policy and
  exception scope. It supplies no graph selection, alias proof or permission to
  move input loads past output stores.
- `constant_float_clamp.py` offers an opt-in IEEE binary32 constant-clamp
  legalization using the existing typed LLVM tokenizer. It accepts finite,
  nonzero, ordered endpoints and only independent maximum/minimum calls within
  one contiguous basic-block group. Other SSA consumers retain their original
  definitions. A shared input NaN guard retains the original operations on its
  NaN path. Input freeze preserves defined values and safely refines undef/poison
  before the guard; strict/constrained FP and environment observations refuse.
  It introduces no numeric policy, instruction-set choice or automatic routing.
- `static_llvm_cfg.py` traverses verified LLVM CFGs with caller-supplied argument
  values, pointer index width and result-free observation policy. It evaluates
  declared-width integer arithmetic and symbolic addresses without memory reads
  or alias assumptions. Unresolved control flow, poison flags and unsupported
  layouts refuse completion; a partial iterator is never a complete count.
  Instruction interpretation and hardware timing rules stay in the provider.
- `segmented_matrix_view.py` proves static nested slices and whole linear
  reshapes as a typed row-segmented matrix address map. It preserves element
  width, exact source endpoints and the original tensor owner without IR edits.
  Consumers must separately close dense physical allocation, read-only access,
  source lifetime and ABI acceptance before eliminating a materialization.
- `codegen.py`, `toolchain.py`, `weights_pack.py` (manifest/safetensors → blob + arg table), `abi.py` (`_mlir_ciface_forward` host runner + `ScalarArg`), `lower.py`/`cli.py`.
- `source_numeric_capability.py` emits separately admitted compiler builtins for
  one numeric component. Its optional finite binary32 floor retains the original
  nonfinite library call, signed zero and single operand evaluation. Standard
  values, interposition, errno and exception observations are explicit; source
  rounding mode is unchanged. Target legality and profitability require the
  provider's actual compilation and execution receipts.
- `kernel_backend.py` — compile one outlined kernel func in isolation + check it vs a numpy reference (the per-kernel bisection harness; used by `runtime.dispatch_runtime`).
- `declaration_access.py` repairs printer-dropped bufferization attributes for
  explicitly named private declarations. Callers supply positional access policy;
  this owner does not import a target rewrite or assume its dtype/signature record.
  It preserves the existing simple declaration syntax, not a general MLIR parser.
  Device and matrix-unit file rewrites share it and refuse unpatched declarations.
- `int8_contractions.py` owns structural signed-int8/int32 contraction outlining,
  declaration emission and signature sidecars. Callers supply selection, symbol
  prefix and sidecar filename explicitly; no target ABI names are defaults.
  Both the outlined CPU backend and the legacy matrix route use this owner.
  The matrix ABI and shim are owned by the selected OOT provider's
  `matrix_lowering` plugin; they are not shared-core imports. Core kernel-emitter
  and certification dependencies still need migration; the shim move does not
  qualify a native route.
- `cpu_bf16_matmul.py` exposes the explicit CPUBlas fallback BF16 four-partial
  reduction, including tails assigned to partial zero. The caller proves actual
  source dispatch and finite/RNE scope; generic matmul is not rewritten by default.
- `cpu_softmax.py` is an explicitly selected Torch2.10 f32 numerical policy: SLEEF
  exp_u10, eight-lane ordered reductions, tail preservation and reciprocal multiplication.
  It requires finite inputs and static last-axis width at least eight; defaults are untouched.
- `cpu_sum.py` provides the explicit AVX2 f32 cascade order for contiguous source
  sums with widths from 8 through 8191. Source dispatch, finite intermediates and
  reassociation permission belong to the caller; generic reductions stay unchanged.
- `requantization.py` owns complete signed-i8 monotone transition proofs for
  ordered binary32 scale chains, optional constant channel bias, and explicit
  local error bounds. `integer_readout.py` generates CPU readout from those
  proofs, including an exact neighboring-threshold correction to a bounded
  fixed-point estimate. Optional early saturation guards use the same proven
  transition thresholds. Independent scalar packets retain each load/store
  pair in source order and bounded tails; no pointer nonoverlap is inferred.
  No accelerator scale/store capability is assumed.
- `quantized_affine_joint.py` proves two byte predictor contracts jointly over
  every signed-i8 input pair. Any tuple requiring different source outputs
  refuses an exact decoder. Its portable in-place decoder consumes independently
  qualified, nonoverlapping predictions of unchanged inputs; provider arithmetic,
  buffer lifetimes and the cost of both readouts remain explicit obligations.
  Its optional complete-domain first-output guard skips second-buffer reads only
  when the first prediction is already proved equal to the source. Conservative
  accepted values are permitted; observed runtime ranges grant no eligibility.
  The default emitted decoder stays unchanged, and neither target store is
  removed by this portable decoder optimization.
- `quantized_affine_bracket.py` derives corresponding-output scale intervals
  from the complete signed-byte source relation. Raw-sum collisions refuse;
  disjoint interval witnesses bound the number of scales within this family.
  A complete joint certificate independently validates the selected decoder.
  Sharing an integer producer, target rounding and workspace lifetime belong
  to an explicitly qualified provider; interval coverage is not a cycle model.
- `guarded_quantized_mean.py` owns a generic signed-i8 Q/DQ mean certificate and
  CPU code generation for contiguous or packed NHWC input. Ambiguous sums replay
  the original ordered float operations; static count1..128, little-endian packed
  word layout, lane bounds and unaligned fallback are explicit contracts.
- `position_table.py` builds an explicit bounded integer-position lookup into a caller-owned
  f32 table. The caller proves the inclusive position interval and binds the table to its source
  numerical policy; there is no automatic transcendental substitution or backend inference.
- `compact_abi.py` — opt-in all-pointer entry compaction from an explicit compiler/runtime
  base-buffer layout. Requires target-derived pointer index widths, a distinct entry symbol,
  complete per-argument bindings, and verifies the whole CFG under pointer substitution.
  It does not infer arena reuse, alignment, no-alias facts, or compatibility with an old harness.
- `impr_features.py` + the per-feature modules next to it (`selfcopy.py`, `transpose_fuse.py`,
  `epilogue_fusion.py`) — NAMED, default-off edits to the pass list / transform schedule. A feature
  defines its own edit and registers itself; the empty feature set must leave the pipeline
  byte-identical. `epilogue_fusion.py` fuses a per-output epilogue (the int8 requant) into the loop
  nest of the reduction that produced it, via affine producer-consumer fusion at zero compute
  tolerance. `requant_fuse.py` does the same job for a contraction the per-op schedule has already
  tiled and vectorized (where the affine fusion is inert): it tiles the epilogue on TENSORS and fuses
  the contraction and its accumulator fill into that tile loop, so the model-sized i32 accumulator is
  never built. Two registered points, because they differ in kind — the plain one only removes the
  traversal, the `_vec` one also reshapes the epilogue tile — and the emitted-code evidence separates
  them.
- `custom_isa.py` — `merlin.inline_asm` → `llvm.inline_asm` 1:1 (custom ISA / `.insn` raw encodings; no LLVM fork). `passes_xdsl.lower_bf16_matmul_f32acc` rewrites bf16 matmuls to accumulate in f32.

## What does not belong here

- Hand-written kernels (`merlin/runtime/baremetal/spike/`), command-buffer pipeline (`xdsl_dialects/lowering/`), model capture (model2MLIR).

## Invariants

- **Accelerator-independent.** Shared passes, weights packing, ABI and runners do
  not branch on accelerator identity. CPU instruction selection belongs in host
  codegen, under an explicit host ISA policy (`codegen.py` flags or the late
  legalization callback); paired portable LLVM provides native verification.
  Target-specific dialects, device instruction encodings/kernels, hardware facts
  and ABI glue belong in the selected OOT compiler backend. Reusable host code
  generation, packing, requantization and runtime stay in Merlin. A vectorization
  stage must be a parameterized compiler choice, not a model or accelerator-name fork.
- `buffer-results-to-out-params` MUST include `modify-public-functions hoist-static-allocs` — otherwise it silently skips the public `@forward` and the entry returns heap-allocated descriptors.
- `quant_ext.*` parses as `builtin.unregistered`: match `op.op_name.data`, not `op.name`.
- Weight tensors are never embedded in C arrays — pointers into the safetensors payload blob, offsets straight from the header (`weights_pack.pack`).
- Vectorization is clang `-O2 -march=rv64gcv` auto-vectorization (verified: emits vsetvli). A scalable-vector tile/vectorize MLIR path may be layered later.
- Host (x86 ctypes) parity vs torch reference is the gate before any spike run.
- `HostModel.load` defaults to `RTLD_LOCAL`, including the >1024-arg trampoline path: the trampoline receives the loaded library's exact entry address. Several model/kernel `.so`s must coexist without their shared `forward`/`memrefCopy` symbols clashing. `emit_c_interface` wraps only memref args as descriptor pointers; scalar args are passed by value — use `abi.ScalarArg` (the dispatch runtime relies on this for `cumsum`-style kernels).

## Testing expectations

- radix_integer_reconstruct.py emits explicitly selected signed-i64 weighted
  group updates followed by one exact binary64 conversion. It rederives the
  canonical i32 group and binary64 prefix proof, requires original RNE/+0
  semantics, and uses defined multiplication for negative terms. A separate
  owned nonoverlapping scratch buffer is mandatory; initialization, traffic,
  conversion and live memory must be measured. No automatic routing is added.

- `encoded_i8_zeros.py` owns target-independent complete nonzero summaries of
  actual encoded signed bytes, optional fused producer row joins and independent
  storage/metadata revalidation. Consumer-supplied panel extents and explicit
  immutable input/metadata plus nonoverlapping output ownership are required.
  Source floating dtype and previous inputs cannot prove encoded zeros. Target
  omission, resources and complete destination initialization remain in OOT.
  The encoded_radix_reconstruct.py companion optionally skips proved
  positive-zero weighted integer updates under a rederived canonical
  i32/binary64 radix bound, explicit RNE and positive-zero prefix seed. Arbitrary
  floating zero elimination is not authorized; producer lifetime/output
  completeness proofs remain necessary.

- `quantized_affine_pair.py` derives complete signed-byte source/predictor
  certificates and explicit C correction. It preserves ordered source binary32
  arithmetic, refuses changed certificates, and leaves runtime ambiguity costs
  unknown. Input preservation, nonoverlapping input and output storage, floating environment and
  independently proved target prediction are caller/provider obligations. No
  model rewrite or default performance policy is enabled by this utility.
  Its explicit `sparse_pair_limit` optionally replaces the correction bitmap
  with grouped operand equality predicates within a caller-supplied bounded
  cardinality. Complete-domain enumeration proves predicate identity, including
  empty and multiple-pair relations. Ordered floating replay is unchanged;
  actual load scheduling and full producer/correction cost require qualification.

`merlin/python/tests/test_llvmlower.py` — synthetic slice e2e (host execution vs Python reference); toolchain-gated tests auto-skip when clang/m2m venv are absent.

## Notes for future agents

Tools: torch-mlir wheel python = full upstream pass registry + translate; clang-23 from `/path/to/merlin-iree/...` targets riscv64 with `+v`. The 27k-line full model goes through the venv pipeline as text — expect minutes, not seconds.

- `insert_slice_destination.py` is an opt-in exact preparation rewrite for a
  sole-use pure pointwise producer inserted into a static splat destination. It
  clones the scalar body into a fresh filled slice, rejects output-init reads and
  index-sensitive bodies, and leaves alias legality to upstream bufferization.
  It does not promise that all destination copies disappear.

- `fresh_tensor_writer.py` is an explicit, target-independent rewrite for bodyless
  full-writing tensor calls. Callers supply result identity, every full writer,
  borrowed symbol and allocation alignment. It validates the entire selection
  before mutation and gives upstream bufferization ownership of fresh buffers.
  ABI bridge emission stays with the provider. No default routing changes.
  Its default declaration ABI remains ranked C. An explicit expanded-memref
  contract may select a distinct wrapper symbol and retarget tensor calls,
  preserving the provider's original raw C entry name. The provider still emits
  the borrowed bridge; matching argument types are never an identity proof.
- `segmented_input_acceptance.py` binds explicitly accepted read-only static
  matrix views to their original typed owners. It validates producer and
  consumer contracts with the existing fresh-writer rewrite on a clone, checks
  every source address map, and preserves numerical attributes and other call
  operands. The provider compiles the accepted map without input writes or
  retention and subsequently runs fresh-writer conversion. It emits no ABI
  bridge, removes no view operations, and changes no default routing.
- `DeviceRouting.post_offload_transform` is an explicit host ABI preparation hook
  after source-bound declaration rewriting. The routing sidecar is immutable,
  selected IR bytes are recorded, and absent hooks perform no I/O or mutation.
  Dense adapter emission reports returned-argument identity only when the caller
  explicitly provides complete kernel output writes; emitted C is unchanged.
  The writer contract can explicitly allow initialized destinations when the
  declaration is write-only and every written element is proven overwritten.
  Fresh allocation then preserves live original destination tensors; the default
  sole-use empty-tensor restriction remains unchanged.

- `bounded_rne_maps.py` proves complete typed bounded ties-even scalar bodies
  and optionally stripmines static parallel tensor maps through proved input
  permutations. It retains live destination tensors through ordinary upstream
  bufferization, with separate exact tails and no inferred pointer no-alias.
- `bounded_rne_packet_llvm.py` recognizes pure straight-line multi-result scalar
  helpers by their complete arithmetic and SSA dependencies, then groups explicit
  CPU RNE operations under the selected host ISA policy. Absent that policy the
  source is unchanged; memory/control flow/strict FP refuse legalization.

- `ordered_fma_groups.py` inventories live typed contraction consumer DAGs and
  their BF16 endpoints, retaining every intermediate source operation and an
  immutable mutation witness. Live float escapes and unsupported effects refuse
  closure. Source closure grants no numerical, alias, scheduling or external-call
  purity permission; providers must prove the complete endpoint separately.

- `quantized_consumer_frontier.py` retains the complete typed consumer DAG from
  explicit floating producers to an integer observation, including every live
  floating scale or residual escape. Static assemblies require complete disjoint
  ownership. Source, uses, coordinates and numeric context are immutable witnesses;
  this analysis grants no approximation, effect or buffer permission.
- `consumer_observed_group_writer.py` requires explicit complete live consumer
  witnesses before binding observationally equivalent group writers. It pins
  every escaping observation, numeric and effect proof, source use and context;
  the original typed consumer remains in the caller. The wrapper verifies source
  coverage and delegates the existing writer ABI and buffer proofs. It does not
  establish the supplied numerical theorem or admit a performance policy.

- `ordered_fma_group_outline.py` is an explicit source-exact partition seam. It
  validates complete closed endpoints and domination before moving original
  operations into an ordinary function, retaining source function attributes.
  Only proved unread pointwise initializers are omitted; reductions and live
  source inputs remain. Exact constant tensors are localized without changing
  their other uses. External numeric providers/ABIs/full writes remain separate
  proofs; ordinary upstream owns buffers and compilation.

- `ordered_bf16_group_binding.py` provides an explicit normal preparation callback
  plus source call/body/ABI coverage checks. It derives policy from typed source
  closure and uses hashes/ordinals only to bind instances. Its ordinary source
  functions remain CPU work; no numeric certificate or target writer is enabled.

- `closed_group_writer.py` binds an explicitly supplied complete-source endpoint
  and effect contract to an ordinary fresh borrowed writer. It validates the
  retained function/context/ABI and complete typed semantic fingerprint before
  mutation. Numerical witnesses remain caller proofs; installation itself grants
  no target coverage or performance qualification. Provenance and function names
  do not choose policy, and equal shape alone never permits physical sharing.

### Source-ordered attention certificate executor

`source_attention_frontier.py` emits a portable, optional host executor from an
explicit current-source plan. It is not a source matcher or routing policy.
Caller admission must already close the complete source arithmetic DAG and the
row-quant consumer's complete observation frontier. The opaque product callback
must have a separately proved signed-radix implementation; the executor does not
trust shape equality as a numerical proof. Target instructions and descriptor
ABI adaptation remain provider-owned.

All mutable state resides in one caller-owned workspace. Query the actual
compiled byte/alignment requirements, accept a larger reusable capacity, prove
input/output/workspace ownership and lifetime, and retain original source
fallback for refusal. Only success publishes the public destination. Tests:
`merlin/tests/runtime/test_source_attention_frontier.py` covers original-order
replay, dirty repeated calls, strided inputs, masked rows, callback refusal,
nonfinite inputs, invalid shape/stride/alignment/capacity, and numeric-plan bounds.

The explicit `prepare_zero_error_blocks` option prepares the original zero-term
outward sum only after full admitted representation-error metadata proves the
block eligible. It preserves source-FMA rounding, checked fallback, and default
bytes; it does not install a routing decision or a query preparation cache.

`prepare_product_domain` is an explicit optional source-attention executor
choice. It derives row/column envelopes from the current private admitted norms
and reconstructed spans; no measured samples or source identifiers choose it.
The exact product callback proof, full-span ownership and stable monotone
outward numeric capability are prerequisites. Uncovered rows use the unchanged
checked path. This feature neither shares preparation across calls nor enables
normal provider routing automatically.
