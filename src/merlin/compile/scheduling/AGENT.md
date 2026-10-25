# AGENT.md — merlin/python/merlin/compile/scheduling

## Purpose

Scheduling passes that sit between "which tile fits" (`compile/capacity.py`) and a backend's
instruction packing: the ORDER block moves and computes are issued in, which operand stays resident,
and where each operand's region sits on chip.

## Modules

- `block_schedule.py` — Order the block moves and computes of a contraction over a target's on-chip
  stores.
- `derive.py` — Derive a block-schedule Geometry from a target's facts.
- `staged_dense.py` / `staged_dense_verification.py` — finite staged descriptor/consume proposals and independent source-product/state checks. Descriptor issue is not synchronous completion; the original target must bind actual emitted effects and independently check final executions.
- `periodic_dense.py` — counted two-window raw dense proposals with complete piecewise affine macro checks. Its new V9 source owner selection is prospective; old inventories and frozen exports stay unchanged. Target emission/completion and execution qualification remain independent.
- `zero_initializer_groups.py` — closed source zero-splat/contraction grouping proposals. These preserve all source operations, have no target authority and are not added to any existing frozen SDK release.
- `staged_rhs_residency.py` — full-K RHS stripe proposals within explicitly assigned private rows. Complete source/product/state checks, a new independent target ownership/placement capability and final actual emitted-address checks remain required; physical capacity alone supplies no row assignment. Existing frozen SDKs stay unchanged.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

- Contract resolution, compact dense, row/schedule compilation and final checking
  share `library_sources.shared_library_sources` (reexported by `schedule_pipeline`). Add semantic/checker helpers
  to that roster rather than creating a per-family version list. Source identities
  neither grant candidate import access nor establish transitive dependency closure.
- `export.export_shared_scheduling` copies the declared core for explicit host
  review, with three minimal namespace initializers. Keep algorithm/checker
  files identical and test all entry families in isolation. New public APIs and
  source membership require explicit review; never infer grants from an export.
- `pointwise_source` checks a current typed tensor body against original target
  calls, then independently reconstructs final decoded scalar dataflow after
  scheduling. Reuse `dense_ir.checked_static_linalg_shell`, `scalar_graph` and
  the existing interval memory. Unknown arithmetic/effects, init reads, shifted
  external aliases and asynchronous arithmetic keep the original source route.
  This source checker does not supply target instruction selection, ABI glue,
  runtime ownership or independent execution qualification.
- Optional `PointwiseScheduleRewrite` implementations receive current typed
  scalar source and selected resource/capability data, then propose actual calls
  and exact private objects. Fixed source, storage, persistent-state and final
  actual-byte checks run independently of the implementation. This bounded
  synchronous integer family cannot admit pulses, unknown state visibility,
  asynchronous arithmetic or numerical policy changes. Pass names/receipts do
  not bind candidate code or create sandbox grants; the original revision and
  execution owners must do so. Directly constructed readout instructions still
  obey the selected RTL capacity even when bypassing a physical emitter.
- **Knobs, not constants.** Load on index change, lookahead depth, operand order/grouping, region
  placement and loop order were each measured or shown to move cycles, and none has a value that wins
  everywhere: the best setting depends on the SHAPE and on the SUBSTRATE. Defaults are documented in
  `Knobs`; a target picks values by measurement. Never bake one into a backend by hand -- mint it
  (`merlin/experiments/scheduled_backend_v0`).
- **Geometry is derived, never passed as a literal.** `derive.geometry_from_address_space` reads block
  edge, store rows, per-bank depth and accumulator rows from `targetgen.address_space`. Stores are
  resolved to roles by ROW WIDTH (`address_space.operand_store`), and an accumulator is scheduled into
  only when `address_space.accumulator_kind` says it is ADDRESSABLE -- an in-datapath accumulator has no
  rows, and its depth is not in the facts. A quantity the facts cannot answer is a `BlockScheduleError`.
- **`block_schedule.py` imports nothing from merlin.** Generated backends may not depend on merlin at run
  time, so they vendor this module byte for byte; the package integrity scan rejects a merlin import.
  Anything that needs merlin goes in `derive.py`. Pinned by `test_block_schedule.py`.
- **One nest for every contraction.** Matmul and convolution share `_schedule_nest`; a convolution is a
  contraction whose streamed blocks are GATHERED (per-row DRAM sources, `None` = the target's zero path).
  A load is due the first time a compute needs it (per nest with `load_on_index_change`, per emission
  group without) -- no loop is named as the one whose index change triggers it, so every `loop_order`
  is correct.
- **Every schedule is checked before it is returned.** `check_residency` refuses a schedule whose load
  overwrites a block a later compute still reads. The block identity includes its gather: the same block
  position with different gathered pixels is different bytes. This is not defensive: when two regions
  overlap, one emission order is correct and another silently computes on the wrong bytes.
- **Every schedule is executable.** `execute` runs the ops at the addresses they name on host arrays;
  the result must equal `lhs @ weight` (or the direct convolution). That is the pass's own oracle.
- **Pooling is not modelled.** It drains the accumulator N-major and defers every drain past the nest --
  a different accumulator layout, refused rather than approximated.
- **Not part of the `merlin-compile` facade.** This package's vocabulary is deliberately generic
  (`Load`, `Store`, `Geometry`); callers import `merlin.compile.scheduling`.

## Testing expectations

- `merlin/tests/infra/test_block_schedule.py` — knob semantics over two synthetic geometries with
  neutral store names, every loop order and conv shape executed exactly, the refusals, mutation controls
  on the residency check and the executor, and that the module stays vendorable.
- `merlin/tests/targetgen/test_store_roles.py` — role resolution, accumulator kind and declared counts,
  on synthetic facts and on the real targets.
- `merlin/tests/gemmini/test_block_schedule_matches_packages.py` — the pass reproduces each measured
  package variant's stream (matmul and the zero-path conv), held to a checked-in golden so it runs on any
  checkout.
- `merlin/tests/gemmini/test_scheduled_package.py` — packages MINTED from the pass emit the hand-written
  packages' raw instruction streams, op for op.


The explicit `DenseBiasSource` path initializes each counted output tile from
a contiguous signed accumulator-width per-column vector, broadcast across rows.
`InitializeAccumulator` declares its actual source columns, strides, physical
rows and extents. The compact checker proves exact initialization coverage,
first-compute accumulation and latest-writer continuity through final stores.
Only signed modular accumulator addition is admitted; arbitrary seeds, scalar
column replication, shifted layouts and hidden readout memory remain unsupported.
These are compiler-side conditional checks, not target execution certification.
The target must independently establish bit-preserving initialization and
synchronous completion; zero-seeded behavior remains the original default.

`plan_spatial_max_readout` is a finite, separately typed spatial drain. It
retains the complete raw output in private column-block-major accumulator rows,
rechecks reduction lifetimes, and checks every pooled window and output channel.
It does not turn pooling into a unary scalar readout. The caller binds the
scalar identity and proves target address/configuration/completion semantics.
Explicit source padding values are in the post-readout signed integer domain;
unknown target padding must refuse. Capacity and window-enumeration budgets are
explicit; this is not a scalable retained-output strategy for arbitrary models.

`window_transfer.CountedConvolutionPlan` partitions the original NHWC convolution
into counted output-row/channel/reduction loops with at most eight static tail
families. Every output tile consumes kh/kw/channel in source order and stores
once. Interior input runs are affine DMA rectangles; boundary rows select a
checked source transfer or explicit device zero transfer. The supplied private
rows must fit and remain disjoint. Synchronous completion is a target assertion
requiring independent evidence; it is not inferred from callback invocation.
The shared emitter performs index/control work only, never payload packing.

`compact_loaded_sum.verify_compact_loaded_sum` checks a narrow counted two-loop
family with exactly three actual projected effects per tile: initialize from
the first input, accumulate the second input, then store the complete live
rectangle. Clipped row/column intervals establish full source/output coverage
including tails without enumerating dynamic tiles. Private accumulator capacity,
row separation, signed index bounds and synchronous completion are explicit.
The caller supplies the exact original scalar proof and its immutable payload
bindings; the checker binds those identities but does not prove target arithmetic,
physical ownership or completion implementations. Unknown schedules refuse.
This prospective API is listed in V8 only; frozen V6/V7 inventories stay fixed.

`spatial_gather.plan_gathered_spatial_max` constructs counted gather/store loops
for a standalone spatial maximum with explicit source geometry and channel tails.
Minimum-valued padding may become duplicated valid border entries only when
every window contains a valid entry. The physical adapter still owns local
readout semantics, storage capacity and synchronous completion. The returned
kernel is consumed by the ordinary index emitter; it is not execution evidence.

`DenseTiledIntegerReadout` retains every original binary32 column coefficient
and requires constant coefficients within the explicitly selected column tile.
It describes signed-i32 conversion, one binary32 multiplication and nearest-even
saturated signed-i8 readout, with optional ReLU. A `TiledIntegerStore` carries its
actual `readout_tile`; symbolic verification ties that operand to the original
output columns and checks complete reduction, i8 output bytes and accumulator
lifetimes. Immutable coefficient tables are command metadata, never permission
for CPU tensor requantization. Actual configuration, target rounding and
completion still require independent implementation qualification.

`spatial_readout.prove_spatial_readout` independently compares both rounded
source readouts with a proposed single readout over the complete signed-i32
transition partition. Multiplying coefficients alone grants no equivalence.
`spatial_producer` emits counted pure-convolution halo rectangles into private
raw accumulator rows, followed by a local spatial drain. It recomputes overlap,
retains source reduction order and refuses intermediate external observers.
Exact source matching, immutable inputs, private allocation, physical readout
and synchronous completion remain independent caller obligations. No normal
compiler route or frozen SDK gains this prospective V10 API automatically.
The separate `staged_readouts` mode keeps both source quantizers: every Q1
stripe store completes before any identity i8-to-accumulator reload, then Q2
and the local maximum drain before reuse. It requires explicit caller-owned
bounded i8 byte storage, rejects collapsed replacements, and supplies no scratch
pointer allocation, nonalias, target movement or runtime ABI permission.

`scf_commands` is a prospective `merlin.scf_commands.v1` bridge from counted
integer command schedules to existing SCF/arith/func IR. Existing runtime event
types retain SSA dependencies and Stage payloads retain private-row lifetimes.
The bounded unroller invokes xDSL's real SCF transform; emission consumes the
transformed operands rather than rebuilding a schedule from metadata. Explicit
launch handles and wait callbacks are required for asynchronous commands. Events
must close within each iteration; carried events, tensor-payload arithmetic,
persistent configuration annotations and unknown operations refuse. Stage checks
are conservative declared-row hazards, not physical ownership or target timing
proof. No old SDK roster/export gains this leaf or the runtime dialect; plain
candidate-local copies remain explicit authored code until a reviewed release.
Prospective inventory V11 extends V10 with the spatial scratch ABI, ordinary
fresh writer/private pool and the pure borrowed-effect premises. V6–V10 retain
their member lists. The pure `borrowed_effects` leaf avoids exporting host native
verification orchestration; `borrowed_buffer_calls` keeps compatibility exports.
No source/effect permission or target qualification follows from library review.

`group_scf_command_loops` partitions selected innermost command loops into
bounded issue regions and a separate remainder while retaining original order
and operands. A contract hash is conditional input, never qualification. Both
real target entry/completion hooks are mandatory when emitting marked regions;
the exit hook runs after the counted loop, before subsequent effects. Queue
issue or entry release is not completion. Source/ownership/descriptor/queue and
accumulator independence need a separate adapter contract. The standard unroller
refuses marked groups rather than erasing their completion boundary.
Its explicit `preserve_issue_groups` policy may instead unroll independent
loops while leaving marked regions, their contents and enclosing loops intact.
This permits ordering real control transforms around target grouping without
dropping completion hooks or treating a queue entry as completed work.

`tile_load_coalescing.coalesce_tile_load_loops` consumes counted primitive-load
loops after their source checks. Explicit target codecs must decode actual
source-byte and local tile-row mappings; source pitch and local block stride
are independent. The rewrite retains partial/nonaffine or asynchronous routes
and never moves a copy across another effect. `expand_tile_load_mapping`
exposes every band of the final actual wide transfer for subsequent hazard
checking; it does not transfer an earlier epoch proof to a new access range.
Decoded hardware bounds and independent target execution remain required.
Historical library inventories and frozen exports do not gain this API.
