# iteration_notes.md — durable cross-round memory

Format: what changed -> what the verdict said -> next hypothesis.
Round 1 is the build-out round; everything below was measured, not assumed.

## Round 1

### R1.0 — survey + plan
Corpus is 111 public capsule dirs (isa 31, layers 45, model 7, model_slices 30); 83 are
`merlin_iface` v0.1 (exact-int) and 28 are `linalg-on-tensors` (tolerance-float, incl. 7 whole-model
capstones).  `docs/PLAN.md` written before any code.  Arm-4 tooling run first and its results are
what the backend is built on (`docs/merlin_provenance.md`): mesh DIM=16, scratchpad 262144 B /
16384 rows, accumulator 65536 B / 1024 rows, RoCC custom opcode 0x7b, funct3 0x3, 26 legal functs.

### R1.1 — package skeleton, first self-check
Built the xDSL front end (`merlin_iface` IRDL dialect), the `gemmini` target dialect, the plan +
tile schedule and the LLVM emitter.  All 83 `merlin_iface` capsules parse + verify + lower.
Verdict (spike): `A1_mvin_mvout` and `A2_single_tile_matmul` PASS end to end.

### R1.2 — DRAM row pitch is PADDED to a whole tile  **(keep this)**
`SY_contraction_i8_partial` (16x31 @ 31x15) failed L2 with the LAST OUTPUT ROW all zeros and the
earlier rows plausible-but-wrong.  That is the signature of a row-pitch mismatch, and the kernel ABI
says `pointee_layout: row-major, edge tiles zero-padded to a multiple of 16 (DIM)`.  Changed every
DRAM row stride to `ceil(cols/16)*16` elements (`plan.row_pitch`).
Verdict: `SY_contraction_i8_partial`, `A7_edge_padding`, `FT01_movement_tail_17x15` all PASS.
**Do not go back to a tight pitch.**

### R1.3 — the trace gate wants an opening FENCE and a faithful CONFIG_ST
`trace does not open with a FENCE` -> the schedule now emits `fence` before `flush`.
`mode relu declared but no CONFIG_ST sets relu activation` -> CONFIG_ST now always encodes the
declared activation/scale even when the full-width i32 readout bypasses the store path (the CPU
lane materialises the stage in that case).  Both violations cleared.

### R1.4 — the emitted kernel must be single-block  **(keep this)**
Every capsule with a compiler-generated CPU-lane epilogue faulted on spike with an EMPTY console.
Bisected with probes injected into a passing capsule (`A2`):
* a block-structured loop (`llvm.br` / `llvm.cond_br`) -> fault;
* `llvm.icmp` + `llvm.select` -> fault;
* straight-line integer arithmetic -> fine.
So `FnBuilder` now emits a **single-block, fully unrolled, select-free** function and every
min/max/clamp/predicate is sign-mask arithmetic (`max(a,b) = b + ((a-b) & ~((a-b)>>63))`).
Verdict after: `C7_attention_qk_i8`, `GP0_matmul_maxpool_i8`, `GP1_matmul_maxpool_tail_i8` PASS.

### R1.5 — the CPU lane must not WRITE the accelerator's staging buffer  **(keep this)**
`A5_relu_epilogue` / `SY_epilogue_relu` / `SY_elementwise_map_i8_sub_tile` still faulted even at 313
emitted lines, so it was not size.  The one thing they did that the passing CPU-lane capsules did
not was store back into the i32 staging buffer the MVOUT DMA had just written.  Folded the
elementwise stages into the GATHER instead (staging is now write-once, owned by the store DMA).
Verdict: all three PASS.

### R1.6 — bias_add is not an epilogue the program oracle implements
`SY_epilogue_bias_add` -> `spike invocation failed: unsupported epilogue stage 'bias_add'
(have: relu, acc_scale, maxpool)`.  The command buffer is correct (L0/L1 pass); the oracle's own
program builder refuses the stage.  Bias is nonetheless reachable in the kernel: the lowering emits
a RES_PACK for the bias tensor, which makes the ABI pass its pointer, and the schedule DMAs it into
the accumulator rows with a zero DRAM stride before the first COMPUTE (this is `repeating_bias` in
the ISA).  NEXT: re-check this capsule on the verilator tier, which may implement the stage.

### R1.7 — first full sweep: 80 / 94
`agent_selfcheck --capsules all --sim spike`: **80 pass**, 0 declined-by-crash.  The 14 failures
grouped into exactly four causes (see R1.8–R1.11).  Note the graded scope is **94**, not the 111
capsule directories on disk: the float `model_slices` (GC*/GF*) and most `model/` capstones are not
in it; the linalg capsules that ARE graded are the 9 `SY_host_lane_*` / `SY_host_only_*`, `GN0`
and `M3`.

### R1.8 — the certifying tier agrees with the screen  **(important)**
`--sim spike` is a SCREEN and cannot certify.  Re-ran a 10-capsule cross-section on the default sim
(`gsim`, the elaborated-RTL tier): **9 / 10 certified**, the tenth being the bias_add gap.  Every
capsule that passed on spike also certified.  Always finish a round with a default-sim run.

### R1.9 — readout planning: narrow FIRST, then pool  **(keep this)**
`GP0_matmul_maxpool_i8` still reported `mode i8 declared but no MVOUT has i8 readout`, because the
maxpool path was reading the accumulator out full-width.  The store path applies its activation and
float scale only on the NARROWING readout (`AccumulatorScale` feeds the full-width port the raw
accumulator), and `maxpool` commutes with the monotone stages ahead of it, so the readout is now
planned as: store-path stages + narrow readout into a staging buffer of the OUTPUT dtype, then pool
on the CPU lane.  This also makes `acc_scale + maxpool` exact without any float code.
`readout_plan()` in `lowering/schedule.py` is the one place that decides this.

### R1.10 — the float host lane: how far it got, and the wall
Chain of measured verdicts for `SY_host_lane_softmax_bf16` and friends:
1. decline -> `interface_to_target / elaboration_error`.
2. lowering succeeds, buffer declines -> `backend_declined`.
3. buffer names the region's op -> `lanes / protocol_violation`: *"capsule forbids lane 'on_mesh'
   and the emitted stream decodes accelerator instructions"*.  The kernel still had `fence`+`flush`.
4. kernel emits ZERO accelerator instructions -> the `lanes` gate PASSES.
5. remaining wall: `spike invocation failed: movement destination 'out0' declares output dtype
   'bf16', which this harness has no buffer width for (sized: ['i32','i8'])`.  Dropping the
   `output_dtype` attribute does not help — it reads the tensor declaration.
Also learned: `numeric.note` for these says *"correctness is graded by running your artifact on the
RTL oracle and checking it computes the declared operation within tolerance"*, so the CPU-lane
program IS the answer — there is just no command shape to hand it over in.  And
`xdsl_dialects/interface.py` says the runtime's non-matmul vocabulary is `VECTOR_MAP`
(add/mul/identity + relu) and `VREDUCE` (sum) only, so no ABI opcode expresses a softmax/layernorm/
gelu region.  NEXT ROUND: the two `SY_host_lane_contraction_*` capsules ARE matmul-shaped and
`COMMIT`'s `KNOWN_OUTPUT_DTYPES` admits bf16/f32 — try a float RES_PACK + MATMUL_RESIDENT + COMMIT
buffer for those two (honest: that is what the region computes) and see whether the harness can size
a float commit destination.  Keep the zero-instruction kernel either way; that part is settled.

### R1.11 — the other three gaps (all oracle-side, all documented)
* `GB0` / `SY_rank_contraction_batched`: `BATCHED_MATMUL` is correct at L0/L1 but the program oracle
  builds only `RES_PACK + matmuls==commits>=1`.  A rank-3 batch cannot be re-expressed in that shape
  without per-batch slice tensors, and the only derivation mechanism the ABI has
  (`params.im2col_recipes`) needs a rank-4 NHWC source.
* `SY_epilogue_bias_add`: `unsupported epilogue stage 'bias_add' (have: relu, acc_scale, maxpool)`.
  Splitting it into `COMMIT` + whole-op `BIAS_ADD` (in place on the output) was tried and crashed the
  oracle (`not enough values to unpack (expected 2, got 1)`), so the faithful FUSED form ships.  The
  kernel does implement the stage: RES_PACK on the bias puts its pointer in the ABI arg list and the
  schedule DMAs it into the accumulator with a zero DRAM stride before the first COMPUTE.
* `M3_host_island_seam_gemmini`: `FALLBACK_ON_ELIGIBLE_REGION`, 21 admitted regions left on the host.
  The placement rule here (`linalg_reader.place`) only admits `family == contraction` at the mesh
  operand dtype, which finds 2 of 8 regions.  The manifest is evidently broader — next round, read
  the capability manifest instead of deriving the rule from the datapath alone.

### R1.12 — straight-line codegen needs a budget  **(keep this)**
A synthetic `maxpool` over a `[16, 4096]` readout took >120 s and would have produced ~1.6 M
instructions.  Since the kernel must be single-block, the CPU lane cannot roll that into a loop, so
`schedule._check_host_budget` now DECLINES past 120 000 estimated CPU-lane instructions (and
1 000 000 accelerator commands) with the count in the reason.  For reference the largest public
capsule, `SY_geometry_odd_tail_heavy` (196x256x768), emits 23 404 accelerator instructions / 2.7 MB
in 2.2 s.

### R1.13 — generalization probes (no simulator needed)
`--shape-coverage`: `all_covered: true`, `multi_tile_axes_uncovered: []`, nothing collapsed.
Hand-written held-out-style probes that the corpus does NOT contain all lower + lint clean:
37x48x33 with `acc_scale 0.125 + relu`; a `requant`+`relu` i32 epilogue with `requant_shift 3`;
a batch-2 conv with dilation 2, stride 2 and asymmetric padding `[1,0,1,0]`; a rank-3 movement;
a standalone `bias_add`; an `attention_pv`.  The conv probe with a deliberately wrong declared
result extent was REJECTED by the derived-shape cross-check, which is the behaviour that stops a
mis-derived geometry from becoming silent wrong arithmetic.

## Round 2

### R2.0 — where round 1 left it
Official verdict: **80 / 96**.  Failure planes: `backend_declined` 10, `spike` 1, `command_buffer` 2,
`model_execution` 2, `model` 1.  `rtl_checks` was CLEAN except three advisory
`T0.data_movement_reuse` info findings on the `SY_kdepth_*` capsules (4x more MVIN than a
fully-resident schedule needs).  Arm-4 calls re-run first: `check_bijection('gemmini')` reports
**no orphan fields and no orphan routes** (the lever set is complete), `derived_levers` returns the
same seven axes, `load_facts` re-confirms DIM 16 / scratchpad 262144 B / accumulator 65536 B, and
`target_repo.generate_skeleton('gemmini')` was invoked before the first edit.

### R2.1 — the host lane now has a REAL generated program  **(keep this)**
Round 1 emitted an EMPTY kernel for an all-host module.  Added two compiler components:
* `codegen/fpbuilder.py` — branch-free f32 scalar primitives.  `max`/`min` are an IEEE sign-mask
  blend of the two bit patterns (no `fcmp`, no `select`, per R1.4); `exp` is a generated
  2^k * poly(r) sequence whose integer k comes from the 1.5*2^23 magic-number rounding trick (so
  no fp->int op is needed and no libm is linked); `erf` is A&S 7.1.26 over that `exp`; `rsqrt` is
  the reciprocal-sqrt seed plus three Newton steps.  bf16 is loaded/stored by hand (shift-16 in,
  round-half-even out).
* `codegen/host_linalg.py` — a structural lowering of `linalg-on-tensors` to those primitives.
  `linalg.generic` walks the iteration space ITS OWN `indexing_maps` and operand extents imply
  (affine dim/constant/add/mul/mod/div all evaluated), `linalg.reduce` its `dimensions`,
  `tensor.expand_shape`/`collapse_shape`/`insert_slice`/`extract_slice`/`splat`, `linalg.matmul`,
  `linalg.transpose`, `linalg.fill`, `linalg.map`.  Nothing is keyed on an op ORDER or a pattern
  name, so a new float region shape needs no new case.
All 10 host-lane capsules lower + emit; `isa_tools lint` decodes ZERO accelerator instructions in
them, which is what `lanes.forbid: [on_mesh]` requires.

### R2.2 — the command shape a host-lane program can be handed over in  **(measured)**
This target's program oracle builds exactly two shapes.  Measured, in order:
* a whole-op (`VREDUCE`) buffer -> `expected RES_PACK(s) + matmuls==commits>=1, got 0/0/0`;
* `MOVEMENT` at f32 -> `movement destination declares output dtype 'f32', which this harness has
  no buffer width for (sized: ['i32', 'i8'])`;
* `RES_PACK + MATMUL_RESIDENT + COMMIT` at f32 -> ACCEPTED (it then went on to check extents:
  `matmul lhs k=32 != weight k=16`).
So `lowering/host_lane.py` now hands a host-lane program over in the resident-matmul shape: one
`RES_PACK` per interface tensor in DECLARATION order (which is what puts each in the kernel's
argument list under `kernel_abi.arg_order_by_command_shape`), plus two compiler-owned carrier
operands sized from the region's own extents.  `params.host_lane_carrier` says so explicitly.

### R2.3 — a float RESULT cannot be handed back on this target  **(the wall; keep this)**
With that shape the kernel RAN.  Probe: store the flat index `i` as an f32 at every output
position of `SY_host_lane_reduction_f32` (out `[16, 1]`).  The console came back
`OUT Y0 16 1 0 1065353216 1073741824 ...` — the f32 BIT PATTERNS of 0.0, 1.0, 2.0 ... 15.0 — and
the numeric layer reported `max_abs_diff 1097859071.3` against a golden of ~0.256, i.e. it read
my word `1097859072` (the bits of 15.0) AS THE VALUE.  `oracle_runner_contract.yaml` says the same
thing in one line: `OUT <name> <rows> <cols> <v0> ... # row-major INTEGERS`.  So this target's
runner has no float readback encoding, however correctly the value is computed.
TWO FACTS FELL OUT OF THAT PROBE AND BOTH ARE WORTH KEEPING:
1. the readback GATHERS WITH THE PADDED ROW PITCH — the 16 printed values were my stores at flat
   indices 0, 16, 32 ... 240, so `ceil(cols/DIM)*DIM` is right for the host lane too (R1.2);
2. outputs are read back by the name `Y0` (the corpus uses that name for the output of all 110
   capsules), not by the emitter's own naming.
The 10 float host-only capsules therefore DECLINE, with the readback fact as the stated reason.
This is a delivery gap, not a lowering gap: the CPU-lane program exists and is emitted.

### R2.4 — batched matmul: L0/L1 now PASS (was a command_buffer failure)
`MATMUL_RESIDENT` unpacks its lhs as `[M, K]`, so a rank-3 operand is `too many values to unpack`
— that is what failed round 1 at the `command_buffer` plane.  Measured chain:
* `RES_PACK + BATCHED_MATMUL(dst=acc) + COMMIT` -> `KeyError: 'acc_Y0'`; declaring `acc_Y0` in
  `tensors` does not help, so `BATCHED_MATMUL` writes a TENSOR and a COMMIT resolves an
  ACCUMULATOR — the two cannot be chained;
* a lone `BATCHED_MATMUL(dst=Y0)` -> **L0 pass, L1 pass**, L2 `expected RES_PACK(s) +
  matmuls==commits>=1, got 0/0/0`.
Shipped as the lone whole-op: it is the only shape the reference models at rank 3, the arithmetic
is now VERIFIED correct at L0/L1, and the remaining wall is the L2 program builder's shape
whitelist.  A `RES_PACK` in front of it is actively harmful — `kernel_abi`'s resident-matmul row
then binds an argument list with no commit output in it and the kernel loses its `Y0` pointer.

### R2.5 — bias_add: the split form is equivalent and ALSO unbuildable
`COMMIT(epilogue=[]) + BIAS_ADD(src=Y0, bias=B, dst=Y0)` gives **L0 pass, L1 pass, numeric exact
(mismatch_count 0 over 256 elements)** — and L2 crashes `not enough values to unpack (expected 2,
got 1)`, i.e. the program builder unpacks the rank-1 bias as a 2-D extent.  Declaring the bias
`[1, 16]` instead makes the REFERENCE reject it (`bias shape (1, 16) != (16,)`), so the two engines
want different ranks and there is no declaration that satisfies both.  The FUSED form ships (it is
what the interface declares and what the kernel does).  Do not re-try this; it is closed.

### R2.6 — the model capstones
`M2` / `SY_micro_model` fail `lane_report_missing_or_malformed`, and round 1's command buffer for
them was structurally INVALID (empty `commands`, `emit_command_buffer` rc=1), which is the most
likely reading of "missing".  The routing plan now speaks the contract's own lane keys
(`on_mesh` / `scalar_rvv_lane`, from `capsule.schema.json:lanes`) and reports them in
`params.lanes`.  But their operands are i8/i64 and the generated CPU lane has f32/bf16 formats
only, so they DECLINE with that stated reason rather than emitting a kernel that writes nothing.
`M3` additionally reports `21 region(s) this target's capability manifest ADMITS were routed to
the host anyway` — the manifest is broader than `family == contraction at the mesh operand dtype`
and is not readable from the granted contract, so the placement rule is unchanged.
NEXT ROUND, in this order: (a) give `host_linalg` integer element types (i8/i32/i64 loads, stores
and scalar arithmetic through the existing `FnBuilder` integer helpers) — that alone unblocks the
generated program for `M3`, whose output is i8 and therefore READABLE; (b) only then look at the
placement rule.

### R2.7 — end-of-round measurement
`agent_selfcheck --capsules all --sim spike`: **80 / 94, 11 certified on the elaborated-RTL tier,
0 regressions** — the 14 failures are exactly the four causes above.  `--shape-coverage`:
`all_covered: true`, `multi_tile_axes_uncovered: []`, `emitted_work` 31 / 42 / 43 / 43 (nothing
collapsed).  `isa_tools lint` on every emitted artifact: 0 UNKNOWN.  The generated transcendentals
were checked against the same formulas in double precision — exp 3.5e-6 rel, erf 1.4e-6 abs, rsqrt
1.5e-7 rel, log 1.5e-6 abs, all far inside the 0.03125/0.02 capsule tolerance.

### R2.8 — a real bug the held-out-style probes found  **(keep this)**
A hand-written `linalg.generic` gelu whose body multiplies by a constant DEFINED OUTSIDE the region
crashed `_scalar_block` with a `KeyError`: the region evaluator only looked its operands up among
the block arguments.  Every corpus capsule happens to splat its constants into a tensor first, so
no capsule exercised it.  `_scalar_block` now falls back to the outer value map for a captured
operand.  This is the second time a probe the corpus does not contain caught a generalization bug
(the first was the conv derived-shape cross-check in R1.13) — keep writing them.

### R2.9 — where to start next round, in priority order
1. **Integer element types in `codegen/host_linalg.py`.**  i8/i32/i64 loads, stores and scalar
   arithmetic through the `FnBuilder` integer helpers that already exist.  This is what blocks
   `M3_host_island_seam_gemmini`, and `M3` is the one model capstone whose OUTPUT IS i8 and
   therefore readable through this target's integer OUT protocol.  Everything else about M3 (a
   valid buffer, a lane report in the contract's vocabulary) is already in place.
2. **`prov.quantization` as a placement input.**  The whole-model capsules carry
   `prov.quantization = "int8_weight_only"` (or `int8_dyn_act_int8_weight`) on the MODULE; the
   host-only capsules carry no such attribute.  That is the only IR-derivable signal that an f32
   contraction inside a model is INTENDED for the i8 mesh, and it is what would let the routing
   plan honestly report `on_mesh` for `M2` / `SY_micro_model` / `M3`.  Do NOT turn it on before the
   int8 mesh lowering exists: a plan that names a lane the kernel never drives is a false plan and
   the capsules' `expected.instruction_classes` gate would catch it anyway.
3. **`T0.data_movement_reuse`** (advisory, `SY_kdepth_*`): 4x more MVIN than a resident schedule
   needs on a shape that fits on chip.  A correctness-neutral scheduling improvement; the seam is
   `lowering/schedule.py`, which is exactly what `action_catalog.escalation_ladder('spatial.dataflow',
   'gemmini')` names.
4. Do NOT re-open: the float readback (R2.3), the batched COMMIT chain (R2.4) and the bias_add split
   (R2.5) are all closed by measurement.

## Round 3

### R3.0 — where round 2 left it, and what the brief said
Official verdict: **80 / 96**, planes `backend_declined:10, spike:3, model_execution:2, model:1`.
`rtl_checks`: 83 / 83 `ok`, findings only the three advisory `T0.data_movement_reuse` infos.
Arm-4 calls re-run FIRST, before any submission edit, and all four returned non-empty:
`check_bijection('gemmini')` -> `orphan_fields=[] orphan_routes=[] unclassified=[]` (the lever set is
complete); `escalation_ladder('spatial.dataflow','gemmini')` -> one HEURISTIC row naming
`<oot_package>/lowering/`; `rtl_backend.derived_levers(target_profile('gemmini'))` -> the same seven
axes, profile DIM 16 / scratchpad 262144 B / accumulator 65536 B; `target_repo.generate_skeleton
('gemmini')` -> 15 scaffold paths.

### R3.1 — the capability manifest is READABLE after all  **(keep this)**
`contract/capsules/conformance/gemmini.yaml:host_lane.admitted_pairs` states this target's admitted
(family, dtype) pairs outright: **contraction/i8, elementwise_map/i8, movement/i8, reduction/i8**, and
`model/M3_host_island_seam_gemmini/capsule.pytorch.py` says the same in prose ("int8 contractions
because contraction/int8 is the only family-and-dtype this target's capability manifest admits on its
mesh").  So R1.11's open question is closed: the placement rule in `linalg_reader.place` (contraction at
the mesh operand dtype) was already RIGHT, and M3's `FALLBACK_ON_ELIGIBLE_REGION` was not a placement
bug — it was that the backend emitted NO program for M3 at all.

### R3.2 — the mixed-lane lowering (`lowering/model_lane.py`)  **(the round's main build)**
A model is not one lane.  New pass splits a `linalg-on-tensors` function into an ordered list of
SEGMENTS — a mesh contraction, or a run of host ops — and gives every value that crosses a boundary a
DRAM buffer.  Mesh eligibility is (a) the placement rule and (b) whether the tile schedule can express
the op: rank-2 `linalg.matmul` at the mesh operand dtype whose `outs` is a ZERO fill (an `outs` that is
anything else carries an initial value the schedule would silently drop, so it stays on the host).
Interface tensors a host segment touches are made resident so the ABI's resident-matmul row puts them
in the kernel argument list.  M3 now emits `RES_PACK x3 + MATMUL_RESIDENT + COMMIT + MATMUL_RESIDENT +
COMMIT` with the LayerNorm island between them, kernel args
`[arg0..arg4, t1, t0, Y0]`, and `isa_tools lint` decodes 38 instructions, 0 UNKNOWN.

### R3.3 — the CPU lane grew an INTEGER domain
`codegen/host_linalg.py` carried every tensor as f32.  An integer tensor is now carried in the integer
domain (i64 SSA) instead: linalg integer arithmetic is exact and modular in the declared width, and
re-deriving it through f32 would round every value past 2**24 and would not wrap at all.  The two
domains are bridged only where the IR bridges them (`arith.sitofp` -> `llvm.sitofp`; `arith.fptosi` ->
a new `FpBuilder.fptosi` that truncates toward zero out of the IEEE fields, because this LLVM dialect
defines no fptosi op and the kernel is branch-free).  Added the integer scalar ops, `linalg.index`, and
integer `linalg.matmul`.

### R3.4 — a falsy-attribute bug that would have hit a hidden capsule  **(keep this)**
`op.properties.get(k) or op.attributes.get(k)` is WRONG: `IntegerAttr(0)` and an empty `ArrayAttr` are
FALSY in xDSL, so the `or` falls through to the other dict and yields `None` — exactly for the values a
zero accumulator init and an empty static-offset list are made of.  It was silently declining every
`arith.constant 0 : iN` in a region body.  Replaced by `host_linalg.attr_of` at all three sites.  This
is the third generalization bug found by a probe rather than by a capsule.

### R3.5 — every entrypoint must answer the SAME way about the same capsule  **(keep this)**
`qa/tier_state.json` records a per-command digest per capsule.  For `M2_microvit_gemmini` and
`SY_micro_model` the last round's `lower_interface_to_target` digest was `e3b0c44298fc1c14` — the
SHA-256 of the EMPTY STRING.  The command buffer was written and the target MLIR was not, because a
codegen-only `LoweringDeclined` returned 1 with nothing on stdout.  That is the most likely reading of
their `lane_report_missing_or_malformed`.  The linalg path now EMITS the artifact during `lower()` and
falls back to a stated decline if it cannot, so the command buffer, the target MLIR and the artifact
always agree.  M2 went from 0 bytes to a lowered module; M3 and SY_micro_model likewise are non-empty.

### R3.6 — bias_add: the ABI's operand form, and the wall re-measured
The ABI gives `COMMIT` a `bias` OPERAND (`command_buffer_abi.yaml`), which this backend was passing as
an attribute only.  Measured: bias as an operand with `epilogue: []` -> reference does NOT apply it
(L0 fail, max_abs_diff 3), so the stage name is what drives it; bias as an operand AND
`epilogue: [bias_add]` -> **L0 pass, L1 pass, mismatch 0**, L2 still
`unsupported epilogue stage 'bias_add' (have: relu, acc_scale, maxpool)`.  Shipped in the operand+stage
form (contract-faithful, no numeric change).  The L2 wall is unchanged and stays closed.

### R3.7 — the CPU lane refuses from the EXTENTS, not after emitting up to the budget
`M0_small_llama_gemmini` took **46 s** to answer and then declined at the 400 000-element budget,
and `SY_model_tiny_llama` took minutes.  An entrypoint that takes minutes to say "no" reads as a
timeout rather than as a decline, so `codegen/host_linalg.estimate_cost` now walks the op list once
(reading extents off the types) and both linalg paths refuse up front.  M0: 46 s -> 0.55 s; M1:
0.83 s; M2: 0.47 s.  Estimates: M0 75 901 354, M1 11 687 225 106, M2 666 948 elements.

### R3.8 — a decline must be answered in the ENTRYPOINT'S OWN LANGUAGE  **(keep this)**
Corollary of R3.5.  `--convert-iface-to-gemmini` and `--emit-target-artifact` used to print NOTHING
and return 1 when the lowering declined.  They now print a module that carries the reason
(`gemmini.declined`) and no accelerator op, and return 0; the refusal itself stays where the
contract puts it, on the command buffer's `declined` entry.  Empty stdout is indistinguishable from
a crashed tool.

### R3.9 — `tensor.extract`, and the gather that genuinely cannot be lowered
Added `tensor.extract` with a COMPILE-TIME-resolvable index (`static_int` folds constants, loop
indices from `linalg.index`, and integer arithmetic over them).  `M2_microvit_gemmini`'s extract is
DATA-DEPENDENT (an embedding lookup), and the kernel is straight-line and select-free, so that one
is a stated decline rather than a guessed address.  M2 is over the host budget as well (666 948).

### R3.10 — the float readback wall, re-measured with the `base` mechanism  **(closed for good)**
`command_buffer.schema.json:tensors.base` says the program oracle "preloads inputs and reads the
output at these addresses", which reads like a MEMORY readback that could honour a declared f32
dtype.  Probe: `SY_host_lane_reduction_f32` emitted in the carrier command shape with `base` on
every tensor.  The kernel ran; the console came back
`OUT Y0 16 1 79304611 77272593 ... 2143289344 ... DONE` and the numeric layer reported
`max_abs_diff 2143289343.5` against a golden of order 1 — i.e. it read the raw f32 WORDS as the
values, exactly as in R2.3, and `base` changed nothing about the readback encoding.  Two independent
measurements now: this target has no float readback.  The 10 float host-only capsules keep the
stated decline.  DO NOT RE-OPEN.

### R3.11 — end-of-round measurement
* `agent_selfcheck --capsules all --sim spike`: **80 / 94 pass, 11 certified on the elaborated-RTL
  tier, 10 declined, 0 regressions**.  The failing set is byte-identical to the one the round opened
  with: 10 float host-only declines, 2 batched, 1 bias_add, M3.
* `agent_selfcheck --capsules <8 cross-section> --sim gsim` (the certifying tier, per R1.8):
  **7 / 8 certified at L3** — `A2`, `A6_resident_reuse`, `B3_conv2d_im2col_i8`, `B4_conv2d_relu_i8`,
  `C7_attention_qk_i8`, `GC7_conv2d_pad_i8`, `GP0_matmul_maxpool_i8` all pass; the eighth is the
  known bias_add oracle gap.  That set deliberately includes the two capsules this round's edits
  touched (the conv COMMIT operand dict and the bias operand), so the change is clean at L3.
* `--shape-coverage`: `all_covered: true`, `multi_tile_axes_uncovered: []`, work 31 / 42 / 43 / 43.
* `isa_tools lint` on every emitted artifact: 0 UNKNOWN.  `disasm` shows every MVIN/MVOUT DRAM
  operand as `{'kind': 'argbase', 'arg_index': N, 'offset': 0}` — no baked address.
* NOTE on `simjob.py`: `--sim verilator` is rejected for this target ("use 'spike' or 'gsim'"), and
  a `simjob submit --sim gsim` job came back with `ModuleNotFoundError: No module named 'xdsl'`
  from its own isolation copy while the SAME capsules certify through `agent_selfcheck --sim gsim`.
  That is a property of the simjob broker's environment, not of the package.  Use `agent_selfcheck
  --sim gsim` for certification.

### R3.12 — where to start next round, in priority order
1. **The model plane is the only open compiler-side question.**  `M3` now emits a correct
   mixed-lane program (both int8 GEMMs on the mesh, all eight required instruction classes, the
   LayerNorm island on the scalar lane) and the plane's answer did not change by one region.  Two
   things are worth trying before anything else, in this order: (a) find out whether that plane
   invokes the package PER REGION rather than per capsule — if it does, the thing to check is what a
   single-region `merlin_iface` module for M3's GEMM looks like coming out of `merlin-compile`;
   (b) look for a fifth entrypoint or a `components:`/manifest field the model plane needs, since
   `your_artifacts` comes back EMPTY for model capsules while it is populated for every other kind.
2. **Do NOT re-open** (each closed by two or more measurements): the float readback (R2.3 + R3.10),
   the batched COMMIT chain (R2.4), the bias_add split (R2.5 + R3.6), and the placement rule
   (R3.1 — the manifest admits contraction/i8 on the mesh and nothing else).
3. `T0.data_movement_reuse` (advisory, `SY_kdepth_*`): 4x more MVIN than a resident schedule needs.
   Correctness-neutral; the seam is `lowering/schedule.py`.
4. Keep writing held-out-style probes.  Three of the four real generalization bugs found so far came
   from probes the corpus does not contain, and R3.4 (the falsy zero attribute) is the worst kind:
   it silently declined every `arith.constant 0 : iN` in a region body.

### R3.13 — the `<unattributed>` bucket was `__pycache__`  **(keep this)**
`qa/.qa_channel/broker.log` shows every capsule's certificate being `invalidated by <unattributed>
(changed)` each round.  The culprit: `submission/mlir_oot/__pycache__/` — the subdirectory caches
are covered by the `mlir_oot/<subdir>/` component entries, but the TOP-LEVEL one is covered by
nothing, so its digest changed on every run and took every certificate with it.  Fixed twice over:
the CLI shim sets `sys.dont_write_bytecode = True` (no bytecode is written at all now), and
`components:` attributes `mlir_oot/__pycache__/` to all four commands anyway.  Verified: 31 files
under `submission/`, 0 unattributed, 0 `__pycache__` after a run.

## Round 4

### R4.0 — where round 3 left it
Official verdict: **80 / 96**, planes `backend_declined:10, spike:3, model_execution:2, model:1`.
`rtl_checks`: 83 / 83 `ok`, findings only the three advisory `T0.data_movement_reuse` infos on
`SY_kdepth_*`.  Arm-4 calls re-run FIRST, before any submission edit, all non-empty:
`check_bijection('gemmini')` -> `orphan_fields=[] orphan_routes=[] unclassified=[] ladder_errors=[]`;
`escalation_ladder('spatial.dataflow','gemmini')` -> one HEURISTIC row naming `<oot_package>/lowering/`;
`rtl_backend.derived_levers(target_profile('gemmini'))` -> the same seven axes, profile DIM 16 /
scratchpad 262144 B / accumulator 65536 B; `target_repo.generate_skeleton('gemmini')` -> 15 paths;
`rtl.facts.load_facts('gemmini')` -> the 26-entry funct table with its names.

### R4.1 — BATCHED MATMUL: one BLOCK-DIAGONAL contraction  **(the round's main build; +2 capsules)**
`GB0_batched_matmul_i8` / `SY_rank_contraction_batched` failed at the `spike` plane with
`expected RES_PACK(s) + matmuls==commits>=1, got 0/0/0`: the ABI's `native_whole_op` kernel-ABI row
admits only `[ATTENTION_QK, ATTENTION_PV, CONV2D]`, so a lone `BATCHED_MATMUL` falls through to the
resident-matmul row, which cannot bind it.  Round 2 read that as "rank 3 is unexpressible without
per-batch slice tensors" -- but a rank-3 batch IS one 2-D contraction:

    out[b*M + i, j] = sum_k a[b, i, k] * w[b, k, j]
                    = (A* @ W*)[b*M + i, j]

with `W*` the weight stack read as one `[B*K, N]` matrix (its row-major layout, unchanged) and `A*`
the BLOCK-DIAGONAL activation `A*[b*M + i, b*K + k] = a[b, i, k]`, zero off the diagonal blocks.
`A*` is DERIVED, not authored: it is an `im2col_recipe` -- the ABI's one additive derivation
mechanism, materialised identically by reference / simulator / device harness -- over the activation
viewed as `[1, B, M, K]` NHWC, with `kh = B`, `kw = 1`, `ci = K`, `stride = [B+1, 1]`,
`dilation = [-B, 1]`, `padding = 0`.  Tap `t` of output row `b*M + i` reads plane `b*(B+1) - t*B`,
which is `b` when `t == b` and OUTSIDE `[0, B)` for every other tap -- and an out-of-bounds im2col
tap reads ZERO, which is exactly the off-diagonal block.  Derived from the ABI's own
`conv_im2col`/`conv_out_dims` definition, then checked against it for
`B,M,K,N` = (1,4,5,3), (2,16,32,16), (3,5,7,4), (4,3,3,2), (2,1,1,1), (5,2,2,2): every one
reproduces the batched product exactly.  Nothing is capsule-specific -- B, M, K, N are the
operands' own extents -- and a batch whose `B*M x B*K` expansion passes
`BLOCK_DIAGONAL_ELEM_BUDGET` is DECLINED with the count stated.
Verdict: `GB0_batched_matmul_i8` and `SY_rank_contraction_batched` PASS on spike and **both
certify at L3 on `--sim gsim`**.  `lint`: 37 instructions, 0 UNKNOWN, MVOUT count 2 = Mt*Nt.

### R4.2 — bias_add: the third and fourth measurements, both walls
Re-measured, one self-check each:
* `epilogue: ["bias"]` (the ABI's OTHER name for the same stage; `commandbuffer.BIAS_STAGES` has
  both) -> `spike invocation failed: unsupported epilogue stage 'bias' (have: relu, acc_scale,
  maxpool)`.  The spike builder rejects BOTH spellings.
* the split form `COMMIT(epilogue=[]) + BIAS_ADD(src=Y0, bias=B, dst=Y0)` with the bias declared
  rank-1 -> **numeric PASS, exact, mismatch_count 0 over 256 elements**, and spike crashes
  `not enough values to unpack (expected 2, got 1)`; with the bias declared `[1, 16]` the crash
  moves to the OTHER engine -- `command_buffer / structural_invariant_violation:
  ValueError: bias shape (1, 16) != (16,)`.
So the two engines demand DIFFERENT ranks for the same operand and no single declaration satisfies
both, and the fused stage is rejected by name under either spelling.  There is no
augmented-matmul workaround either: folding the bias into the K axis needs a column of ONES (im2col
gathers or zeroes, it cannot synthesise a 1) and the bias is `i32` where the mesh operands are `i8`.
The contract-faithful FUSED form ships.  **Closed by four measurements now; do not re-open.**

### R4.3 — the MODEL plane does not read our command buffer  **(measured; closes R3.12 item 1)**
`M3_host_island_seam_gemmini` reports `21 region(s) ... ADMITS were routed to the host anyway` and the
number did NOT move when round 3 replaced M3's do-nothing lowering with a full mixed-lane program.  So
this round PROBED it directly: `params.lanes` was temporarily rewritten to claim EVERY region on
`on_mesh` and nothing on the host, and the self-check re-run.  The answer came back byte-identical --
same category, same 21.  The self-check also reveals the engine:
`numeric.engine = "merlin-compile model --target gemmini --run mesh --verify"`,
`numeric.measured_on = "host_lane_fallback"`, `your_artifacts = {}`.
So the whole-model plane compiles the model with its OWN model compiler and reports on ITS placement;
our routing plan is not an input to that number.  Two further facts say the placement it is unhappy with
is not one we could fix anyway: the capsule's own `capsule.pytorch.py` states that "the widening cast,
the LayerNorm and the requantize all fall to the host/scalar lane", which is exactly the placement this
backend makes, and 21 is precisely the number of ops carrying `prov.region_id = "layer_norm_0"` -- the
one region the target's manifest demonstrably does NOT admit (`normalization` is absent from
`conformance/gemmini.yaml:host_lane.admitted_pairs`).  Recorded, not fixed; do not re-probe.

### R4.4 — a WHOLE MODEL is not graded through the integer readback  **(+ a real lowering change)**
`M2_microvit_gemmini` and `SY_micro_model` fail `model_execution / lane_report_missing_or_malformed`,
and both were DECLINED by this backend -- M2 over the CPU-lane budget, `SY_micro_model` on R2.3's
"a float result has no readback encoding" rule.  That rule is about the RUNNER's
`OUT <name> <rows> <cols> <integers>` protocol, and R4.3 shows a whole model never goes through it: it
is verified by the model engine.  So `linalg_reader` now derives `whole_model` from the module's own
`prov.weights_file` attribute (exactly the 7 `model/` capsules carry it; no other capsule does) and
`host_lane.build` no longer refuses a whole model on readback grounds.  `SY_micro_model` now emits a
real program with a routing plan (5 commands, 11.5 MB artifact, 15 s) instead of a refusal.  M2 stays
declined: its embedding lookup is a DATA-DEPENDENT `tensor.extract` and its host cost is 666 948
elements against the 400 000 budget -- both real, both stated.

### R4.5 — the standalone `BIAS_ADD` now runs on the MESH  **(generalization, from the ISA header)**
`_perf/PF02_bias_add_m16k16n16` (a `dev`-label capsule of the same whole-op) declares
`expected.instruction_classes = [FLUSH, CONFIG_EX, CONFIG_LD, MVIN, CONFIG_ST, MVOUT]`, and this
backend emitted the op on the CPU lane -- THREE instructions, none of them a MVIN or a MVOUT.  No
graded capsule covers the standalone op, so nothing caught it.  The ABI says what to do
("a target with no separate vector-add class is expected to fold this into its accumulator read-out")
and the shipped ISA header says how: `sp_tiled_matmul_ws` moves a `repeating_bias` in with a ZERO DRAM
row stride to `D_sp_addr_start = 1 << (ADDR_LEN-1)`, and `sp_tiled_resadd` adds the second operand with
a move-in to `3 << (ADDR_LEN-2)` -- the same accumulator address with the accumulate bit set -- then
moves the summed row out.  Both bit positions match `tables/isa.py` exactly (`ACC_ADDR_BIT` bit 31,
`ACC_ACCUMULATE_BIT` bit 30, `ACC_FULL_ROW_BIT` bit 29).  `schedule.bias_add` now emits that per output
tile.  Verified on a `[19, 23]` probe: `FLUSH 1, CONFIG_EX 1, CONFIG_LD 8, MVIN 8, CONFIG_ST 1,
MVOUT 4` -- MVOUT = ceil(19/16)*ceil(23/16) = 4 -- 0 UNKNOWN.  Operands NOT in the accumulator's dtype
(which the ABI says cannot legitimately occur) keep the CPU-lane form rather than being fed to an
accumulator port that does not carry them.

### R4.6 — two contract cross-checks a hidden capsule would have caught
* CONV2D + fused maxpool: the ABI says `pool_in_dims` is the conv's OWN output extent and "a
  disagreement is rejected, never reconciled".  This backend read the attribute and pooled over it, so
  a capsule declaring a `pool_in_dims` the geometry does not support would have been silently pooled
  over the wrong window.  It now DECLINES, naming both extents.
* `epilogue_out_shape`: `pool_in_dims` that does not DIVIDE the committed row count used to fall
  through `m // (ih * iw)` and pool across a batch boundary.  It now declines with both numbers.
* `movement`: the round trip is staged in the operand scratchpad (`i8`) or the accumulator (`i32`),
  and any other container was being moved as whatever the hardware's element width is -- a bf16
  movement came back reinterpreted, which is a wrong answer wearing the shape of a right one.  The
  legal set is now `{i8->i8, i8->i32, i32->i32}` (the third staged through the accumulator with the
  load's `shrunk` bit CLEAR, since that bit says the DRAM side is operand-width), everything else
  declines.  All six corpus movement capsules are `i8 -> i8` or `i8 -> i32`, so nothing regresses.

### R4.7 — held-out-style probes run this round (none of these shapes is in the corpus)
19 probes across both grammars, all four entrypoints, schema-validated, `lint`ed and disassembled:
batched `B` = 1/2/3/4/5 with tails (`3x17x33 @ 3x33x15`, `4x8x8 @ 4x8x40`, `2x9x17 @ 2x17x11` with a
RANK-3 declared result) and with an `acc_scale + relu` epilogue; `requant_shift` with M/K/N tails
(23x37x19); two resident groups with different N and different epilogues; a batch-3 conv with
dilation 2, stride 2, asymmetric padding and a fused maxpool; a 1x1 (pointwise) conv over a batch-2
NHWC input; attention_qk and attention_pv at 17/19/33; a rank-3 movement; an i32->i32 movement; a
standalone bias_add; a 1024-deep K with a fused `bias_add + acc_scale + relu`; 1x1x1; 1x4096x1.
Every one either lowers with `MVOUT == ceil(M/16)*ceil(N/16)`, 0 UNKNOWN and every DMA address an
`argbase` (no baked DRAM address), or DECLINES with the reason and the extents (`batch 512` blows the
block-diagonal budget; a 5x5 kernel over a 3x3 input is an empty extent; bf16 movement has no
container).  The block-diagonal batched gather was additionally checked against the ABI's OWN
`conv_im2col` for every probe: derived shape and derived product both exact.

### R4.8 — two view-conflict guards a shared operand would have needed
The batched lowering re-declares its operands under a different VIEW of the same row-major bytes
(the activation as `[1, B, M, K]` for the gather, the weight stack as `[B*K, N]`).  Two probes the
corpus does not contain found the hazards: a module with TWO batched contractions sharing one
activation read the second one's operand under the FIRST one's rank-4 view and declined for the
wrong reason, and a module that `resident_pack`s a weight before contracting it batched would have
had that weight reshaped underneath the earlier command.  `Builder._reshape` now remembers the shape
each leaf was DECLARED with (so a shared activation is re-read correctly and each contraction
derives its own gather -- the two-batched-matmul probe now emits
`RES_PACK, MATMUL_RESIDENT, COMMIT, EVICT` twice with two recipes) and REFUSES to reshape a tensor an
earlier command already read, naming both shapes.  The derived gather is also named per result
(`<out>_lhs_batchdiag`) rather than per activation, so two contractions over one activation do not
collide.

### R4.9 — end-of-round measurement
* `agent_selfcheck --capsules all --sim spike`: **82 / 94 pass, 12 fail, 0 regressions** (was 80/94
  at the start of the round).  The two new passes are the batched capsules; the remaining 12 are the
  10 float host-only declines, `SY_epilogue_bias_add` and `M3`.
* `agent_selfcheck --capsules GB0_batched_matmul_i8,SY_rank_contraction_batched --sim gsim` (the
  CERTIFYING tier): both **certify at L3**.
* `--shape-coverage`: `all_covered: true`, `multi_tile_axes_uncovered: []`, work 31 / 42 / 43 / 43.
* `probe_all.py` over all 111 capsule directories on disk: **85 lower, 26 decline with a stated
  reason, 0 problems** (85 is one more than round 3: `SY_micro_model`).
* `isa_tools lint` over all 87 emitted artifacts: **74 845 instructions, 0 UNKNOWN**.  `disasm` over
  the same: **17 472 DMA operands, 17 444 `argbase`, 28 compiler-owned staging globals, 0 baked
  constant DRAM addresses**.
* Integrity: an AST scan over every `.py` under `submission/` finds no `merlin` import and no `re`
  import; 31 files, all 31 attributed by `components:`, no `__pycache__`.
* Certification cross-section on `--sim gsim`, chosen to cover every code path this round changed:
  `A1_mvin_mvout`, `A2_single_tile_matmul`, `B3_conv2d_im2col_i8`, `FT01_movement_tail_17x15`,
  `GB0_batched_matmul_i8`, `GP1_matmul_maxpool_tail_i8`, `GP2_conv2d_maxpool_i8`,
  `SY_epilogue_maxpool`, `SY_movement_i8_partial`, `SY_rank_contraction_batched` —
  **10 / 10 certified at L3**, run AFTER the last edit of the round.

### R4.10 — where to start next round, in priority order
1. **Nothing is left on the compiler side of the four remaining failure groups.** Each is closed by
   two or more independent measurements, listed here so a later round does not spend itself
   re-deriving them: the float readback (R2.3 + R3.10 + the bitcast arithmetic in R4's REPORT), the
   bias_add stage rejection AND the split form's rank contradiction (R2.5 + R3.6 + R4.2, four
   measurements across both spellings and both ranks), the model plane's independence from our
   routing plan (R4.3, probed directly), and the placement rule (R3.1, from the capability manifest
   plus M3's own loader prose).
2. The highest-value work left is **generalization for the held-out capsules**, and the way to do it
   is the R4.7 loop: write a probe the corpus does not contain, run all four entrypoints, lint and
   disassemble, and CHECK ANY DERIVED OPERAND against the ABI's own definition of the derivation.
   Four of the five real generalization bugs found so far came from probes, not from capsules.
3. `T0.data_movement_reuse` (advisory, `SY_kdepth_*`): the ISA moves up to `MAX_BLOCK_LEN` tiles per
   MVIN and this schedule moves one, which is the whole factor of 4.  Deliberately NOT taken: it
   rewrites the instruction stream of every certified capsule and buys no correctness.  If a later
   round takes it, re-certify the whole corpus at `--sim gsim`, not just the screen.
4. If `SY_micro_model` still reports `lane_report_missing_or_malformed` after R4.4, that rules out
   "the plane wanted a program" and the remaining hypothesis is that the plane needs a fifth declared
   command; there is room for one (`manifest.schema.json:commands` is `additionalProperties: true`).

## Round 5

### R5.0 — where round 4 left it
Official verdict: **82 / 96**, planes `backend_declined:10, spike:1, model_execution:2, model:1`.
`rtl_checks`: **83 / 83 `ok`**, findings only the three advisory `T0.data_movement_reuse` infos on
`SY_kdepth_*` (no `reject`, no illegal funct, no capacity finding).  `integrity_status: clean`.
`shape_coverage`: `all_covered: true`, `multi_tile_axes_uncovered: []`, work 31 / 42 / 43 / 43.
Arm-4 calls re-run FIRST, before any submission edit, all non-empty: `check_bijection('gemmini')` ->
`orphan_fields=[] orphan_routes=[] unclassified=[] ladder_errors=[]`; `escalation_ladder
('spatial.dataflow','gemmini')` -> one HEURISTIC row naming `<oot_package>/lowering/`;
`derived_levers(target_profile('gemmini'))` -> the same seven axes; `load_facts('gemmini')` -> the
26-entry funct table (custom opcode 123 / funct3 3); `target_repo.generate_skeleton('gemmini')` ->
15 scaffold paths.

### R5.1 — the four closed walls, re-checked against the CONTRACT rather than re-probed
Round 4 closed each of the 11 non-model failures by measurement.  This round checked the two that
cost the most capsules against the contract text itself, which is cheaper than another probe and is
independent evidence:
* **the float readback (10 capsules).**  `oracle_runner_contract.yaml` states the harness output
  format outright -- `OUT <name> <rows> <cols> <v0> ... # row-major integers` -- and closes with
  `correctness: "integer, exact == across all three (reference, simulate, oracle); no FP tolerance"`.
  That is a THIRD independent statement of R2.3 + R3.10's measurements.  The one mechanism that
  looked like it might carry a float back, `tensors.physical` ("physical->logical readback layout the
  backend wrote"), is a LAYOUT permutation (`{"unstack_row_halves": 2}`), not a dtype
  reinterpretation, and its vocabulary appears nowhere in the granted contract -- inventing a key
  for it would be guessing a mechanism, not deriving one.  `tensors.preload_b64` is explicitly
  "not emitted by the package".  The 10 declines stand.
* **`bias_add` (1 capsule).**  The remaining untried idea was a separate post-COMMIT vector add.
  `command_buffer_abi.yaml`'s `opcodes` block has **no vector-map opcode at all** (RES_PACK,
  MATMUL_RESIDENT, MATMUL, COMMIT, EVICT, MOVEMENT, ATTENTION_QK, ATTENTION_PV, BIAS_ADD, CONV2D),
  and `commandbuffer.py` defines no `VECTOR_MAP`, so there is no command to carry one.  Folding the
  bias into the K axis still needs a column of ones that im2col cannot synthesise.  Closed for the
  fifth time.
The model plane gives NO local signal at all: `agent_selfcheck --capsules SY_micro_model` (spike and
gsim both) returns `n_capsules: 0, no_results: true` while a normal capsule returns a verdict in
seconds.  That is consistent with R4.3 (`your_artifacts = {}`, the plane runs its own model
compiler) and means M2 / M3 / SY_micro_model cannot be iterated on locally this round.

### R5.2 — `acc_scale` had NO readout outside the narrowing store path  **(a real coverage bug)**
Found by probing an attribute combination the corpus does not contain.  The public corpus uses
`acc_scale` in exactly one shape -- `output_dtype = "i8"` with a POSITIVE scale -- which is the one
case `readout_plan` sends to the hardware store path.  Every other shape fell through to the CPU
lane, where `_apply_scalar_stages` implemented only `relu` and `requant` and RAISED.  Measured:
`epilogue=["acc_scale"]` with `output_dtype` `i32`, `i16`, or a NEGATIVE scale -> the artifact
entrypoint exited 1 with EMPTY stdout.  This is exactly the holdout the task warns about: an
epilogue that passes every public capsule and fails one that changed only the readout width.
`codegen/llvm_emit._acc_scale` now generates the stage, derived from the ISA header rather than
invented: `gemmini_params.h` defines `ACC_SCALE(x, scale) = clamp(ROUND_NEAR_EVEN((float)x *
scale))`, and `command_buffer_abi.yaml` says the same (`clamp_i8(round_near_even(acc * scale))`).
So the generated readout takes the product in **f32** (the width the hardware's scale unit
multiplies in, so a value past 2**24 loses the same bits on both paths), rounds half-to-even, and
clamps to the target's OWN operand dtype read from the RTL facts -- no hardcoded width, no literal
scale (the value is read off the capsule's attribute).
`readout_plan` is UNCHANGED, so every currently-certified capsule keeps the hardware store path.

### R5.3 — the magic-number rounding constant was wrong, and simulating it caught that
`round_near_even` is branch-free (R1.4), so it is the add/subtract-a-magic-constant trick.  The
first version used 1.5 * 2**23 and blended on |x| >= 2**23.  Simulating the EMITTED op sequence in
numpy float32 against the header's own `ROUND_NEAR_EVEN` over 184 024 values found **5 298
mismatches** (~3%), all for |x| in [2**22, 2**23): with that constant the sum reaches 2**24, where
the f32 ulp is 2 and the fraction is lost.  The constant must be **2**23** (signed like `x`), which
keeps the sum in `[2**23, 2**24)` where the ulp is exactly 1.  Re-checked: **0 mismatches over
184 024 values**, and the whole `ACC_SCALE` stage (sitofp / fmul / round / clamp / fptosi) agrees
with the header's macro on **0 mismatches over 93 600** cases spanning 13 scales including negative
ones, tie values and the full i32 accumulator range.
Lesson worth keeping: a branch-free numeric primitive is worth SIMULATING op-for-op against its
defining formula before it ships -- the wrong constant lints clean, decodes clean and is wrong on
3% of inputs.

### R5.4 — the four entrypoints disagreed about the same capsule  **(keep this)**
The same probe exposed a second, more serious bug.  On a codegen-only decline the merlin_iface path
did what R3.5/R3.8 had already fixed for the linalg path: `emit_command_buffer` wrote a buffer with
four REAL commands and no `declined` entry, `parse` and `lower_interface_to_target` both returned 0
with a full target module, and `emit_target_artifact` returned **1 with empty stdout**.  Three
entrypoints claimed the capsule lowers and the fourth read as a crashed tool -- and R3.5 showed that
an empty artifact digest is the most likely reading of `lane_report_missing_or_malformed`.
Fixed in two places, so it cannot come back through a different stage:
* `schedule._check_host_stages` refuses an epilogue stage the generated CPU-lane readout does not
  implement while the PLAN is being built, so the refusal reaches every entrypoint through the one
  path that already carries declines.  The supported set lives next to the split that decides what
  lands on the CPU lane.
* the CLI's codegen backstop no longer returns 1 with empty stdout: it restates the refusal on the
  command buffer (`declined`, zero commands) AND prints a module carrying the reason, returning 0.
Verified with a forced codegen decline in a throwaway copy of the package: `rc=0`, command buffer
`declined` with 0 commands, stdout a `gemmini.declined` module.

### R5.5 — held-out-style probes run this round (none of these shapes/attributes is in the corpus)
All lower, `lint` 0 UNKNOWN, and every one's `MVOUT` count equals ceil(M/16)*ceil(N/16):
`requant` at 17x33x19 (MVOUT 4) and `["requant","relu"]`; `["relu","acc_scale"]` (the reversed
order -- equal to the store path's scale-then-activation only because the scale is positive, which
`readout_plan` is what checks); `acc_scale` at output i32 / i16 / negative scale;
`attention_qk` with `["acc_scale","relu"]` at 17x19 (MVOUT 4); `attention_pv` with `requant` at
17x33 (MVOUT 6); a batch-2 conv with asymmetric padding `[1,0,1,0]`, `acc_scale` and an **i16**
readout (MVOUT 6); conv + `acc_scale` + fused maxpool; and a TWO-OUTPUT module sharing one resident
weight across two different M and two different epilogues, one taking the store path and one the CPU
lane (MVOUT 8 = 4 + 4).  `disasm` over all of them: **0 baked DRAM addresses** -- every DMA rs1 is
an `argbase` or the address of a compiler-owned staging global.
The conv probe was ALSO rejected first for declaring `tensor<126x11>` where the geometry derives
`[90, 11]` (padding `[1,0,1,0]` over 9x7 gives Ho=9, Wo=5, N=2): the derived-shape cross-check
caught the probe's own arithmetic error, which is the behaviour that stops a mis-derived geometry
from becoming silent wrong arithmetic.

### R5.6 — the placement rule was reconsidered and deliberately NOT broadened
`conformance/gemmini.yaml:host_lane.admitted_pairs` admits FOUR pairs on the mesh --
contraction/i8, elementwise_map/i8, movement/i8, reduction/i8 -- while `linalg_reader.MESH_FAMILIES`
admits only `contraction`.  That looked like M3's `FALLBACK_ON_ELIGIBLE_REGION` defect, so it was
checked properly: in the ISA corpus this target expresses `elementwise_map` and `reduction` ONLY as
EPILOGUE STAGES on a contraction's readout (`SY_elementwise_map_i8_aligned` is a matmul with a
`relu` epilogue; `SY_reduction_i8_aligned` is a matmul with a `maxpool` epilogue), which
`semantic_families` states as well (`acc_readout_scaled -> elementwise_map composed_with
contraction`).  There is no standalone command for either.  Broadening `place()` alone would report
a lane the kernel does not drive -- the false plan R2.9 warns about -- and M3 has no i8 elementwise
region anyway (its casts and LayerNorm are f32), so it would not move M3 by one region.  Left
unchanged, deliberately, and recorded here so a later round does not re-derive it.

### R5.7 — disasm fields reconciled against the command buffer, field by field
The mandated check (workflow item 8): a correctly-NAMED instruction carrying a WRONG FIELD lints
clean and still diverges on hardware.  Decoded the two-output probe (one resident weight, Y0 =
[19, 17] i32 with `relu`, Y1 = [21, 17] i8 with `acc_scale` 0.25) and read every field back against
what the command buffer declares for that same command:
* `CONFIG_LD stride` 48 and 32 == the two operands' PADDED row pitches
  (ceil(33/16)*16 * 1 B and ceil(17/16)*16 * 1 B).
* `CONFIG_ST out_stride_bytes` 128 for Y0 (ceil(17/16)*16 elems * 4 B, i32) and 32 for Y1
  (same pitch * 1 B, i8) -- the pitch tracks the READOUT WIDTH, not the accumulator's.
* `CONFIG_ST acc_act` 1 / `relu: True` for Y0 and 0 for Y1; `acc_scale` 1.0 for Y0 and **0.25** for
  Y1, whose bits (1048576000) are the f32 the capsule declared.  Y0's group encodes the declared
  activation even though its full-width readout applies the relu on the CPU lane (R1.3).
* `MVOUT readout` `i32` for Y0 and `i8` for Y1 == each commit's own `output_dtype`.
* MVOUT `rows`/`cols` per tile: 16+3 rows x 16+1 cols for Y0 and 16+5 x 16+1 for Y1 -- exactly the
  declared 19x17 and 21x17, with the tails carried as short tiles rather than padded writes.
* MVOUT `dram` for Y1: `argbase arg_index=4`, offsets 0 / 16 / 512 / 528 == tile (r, c) at
  `r*16*pitch + c*16` elements with pitch 32 -- and `arg_index 4` is Y1 under the ABI's
  resident_matmul order `[W, A0, A1, Y0, Y1]`.  Y0's MVOUTs address a compiler-owned staging global
  instead, which is what `readout_plan` chose for a non-narrowing readout.
Nothing disagreed.  This is also the check that would catch a store stride or readout dtype the
command buffer cannot carry, which is why it is worth re-running whenever the readout plan changes.

### R5.8 — end-of-round measurement
* `agent_selfcheck --capsules all --sim spike`: **82 / 94 pass, 12 fail, 0 regressions**, 11
  certified on the elaborated-RTL tier.  The failing set is byte-identical to the one the round
  opened with (10 float host-only declines, `SY_epilogue_bias_add`, `M3`).  This round changed no
  pass count and was not expected to: both defects it fixed are unreachable from any public capsule.
* `--shape-coverage`: `all_covered: true`, `multi_tile_axes_uncovered: []`, `n_declined: 0`,
  `n_collapsed: 0`, work 31 / 42 / 43 / 43 (unchanged).
* `probe_all.py` over all 111 capsule directories: **85 lower, 26 decline with a stated reason,
  0 problems** (unchanged from R4.9).
* `isa_tools lint` over all 87 emitted artifacts: **74 845 instructions, 0 UNKNOWN** — byte-for-byte
  the same counts as R4.9, which is the evidence that the certified capsules' instruction streams
  were NOT touched (`readout_plan` was deliberately left alone).  `disasm`: 17 444 `argbase` DMA
  operands, 28 compiler-owned staging globals, **0 baked DRAM addresses**.
* Integrity: AST scan over all 24 `.py` files — no `merlin` import, no `re` import, 0 violations;
  31 files under `submission/`, all 31 attributed by `components:`, no `__pycache__`.
* Schema: the command buffer validates against `command_buffer.schema.json` and `manifest.yaml`
  against `manifest.schema.json`.
* Certification cross-section on `--sim gsim`, chosen to cover every code path this round changed
  (both the store-path and CPU-lane readouts, relu / acc_scale / requant / maxpool):
  `A2_single_tile_matmul`, `A4_acc_scale_i8`, `A5_relu_epilogue`, `B2_linear_acc_scale_relu_i8`,
  `C7_attention_qk_i8`, `GP1_matmul_maxpool_tail_i8`, `SY_elementwise_map_i8_sub_tile`,
  `SY_epilogue_acc_scale`, `SY_epilogue_maxpool`, `SY_epilogue_relu` — **10 / 10 certified at L3**,
  re-run AFTER the last edit of the round (including the doc edits).

### R5.9 — where to start next round, in priority order
1. **Do NOT re-open the four closed walls.**  Each now has three or more independent measurements,
   and round 5 added a CONTRACT-level confirmation to the two expensive ones: the float readback
   (R2.3 + R3.10 + `oracle_runner_contract.yaml`'s "row-major integers" / "integer, exact ==" —
   and `tensors.physical` is a layout map, `tensors.preload_b64` is harness-owned, so neither is a
   way in); `bias_add` (R2.5 + R3.6 + R4.2 + the ABI having no vector-map opcode at all); the model
   plane's independence from our routing plan (R4.3); and the placement rule (R3.1 + R5.6).
2. **The model plane gives NO local signal** — `agent_selfcheck --capsules SY_micro_model` returns
   `n_capsules: 0, no_results: true` on both spike and gsim, while ordinary capsules answer in
   seconds.  So M2 / M3 / SY_micro_model can only be moved by a hypothesis tested through the
   OFFICIAL round.  The remaining untested hypothesis is still R4.10's: the plane may want a fifth
   declared command (`manifest.schema.json:commands` is `additionalProperties: true`).  That is one
   cheap experiment worth spending a round's official feedback on, and it is the only lever left on
   three capsules.
3. **Keep probing attribute combinations the corpus does not contain.**  This is now 6 for 6: every
   real generalization bug found in five rounds came from a probe, not from a capsule, and round 5's
   two (an epilogue stage with no readout outside one declared width; four entrypoints contradicting
   each other on a codegen-only decline) were both invisible to all 96 public capsules.  The corpus
   uses `requant` ZERO times and `acc_scale` in ONE readout shape, so the attribute surface is much
   wider than the corpus samples.  Untested combinations remaining: an epilogue whose `maxpool`
   `pool_padding` is nonzero (needs `pool_pad_value`), `output_dtype` `i64` through a pooled readout,
   and a conv whose `dilation` makes the derived extent smaller than the pool window.
4. **Simulate any new branch-free numeric primitive op-for-op against its defining formula** before
   shipping it (R5.3).  The wrong magic constant lints clean, decodes clean, certifies on every
   public capsule, and is wrong on 3% of inputs.
5. `T0.data_movement_reuse` (advisory, `SY_kdepth_*`) is still deliberately NOT taken: it rewrites
   the instruction stream of every certified capsule and buys no correctness.

### Post-freeze correction — fused-bias kernel ABI
The September 5 frozen submission is evidence and remains unchanged.  Under the current ABI contract,
its synthetic `RES_PACK B -> B_res` incorrectly makes `B` look like a resident matrix weight, yielding
`[W, B, A0, Y0]`, while the runner-owned harness correctly calls `[W, A0, Y0, B]`.  The earlier notes
above describe that historical implementation; they are retained rather than silently rewritten.

This fork removes the synthetic bias `RES_PACK` and resolves the declared `COMMIT` bias through the
ABI's trailing `commit_biases_group_major` block.  It also refuses a bias stage that names no tensor.
The resulting command buffer and target module both resolve `SY_epilogue_bias_add` as
`[W, A0, Y0, B]`; bias is still DMA'd once into the accumulator with zero row stride before compute.

### Post-freeze development — capability-selected native convolution

The public `merlin_iface.conv2d` route now forms a canonical convolution task when—and only when—a
pure selector proves the target capability, i8 narrow semantics, NHWC/flattened-HWIO physical
layout, uniform representable geometry, safe i32 accumulation, and exact buffer shapes/strides.
The scheduler tiles this task with the public `tiled_conv_stride_auto` capacity policy and emits a
complete CONFIG_1..6 + LOOP_CONV_WS descriptor for every tile. No model name or fixed layer shape
appears in the implementation.

Two local Spike probes are exact after final codegen: a stride-2/padding-1 3x3 convolution (one
descriptor, 192 outputs) and a CI=1024 case that forces two reduction descriptors (144 outputs).
The latter is the accumulator-continuity test: only the final K slice receives a non-null output
pointer, matching `tiled_conv`.

The existing full-width interface conv fallback is byte-identical to the predecessor in both its
command buffer and target LLVM. Full ResNet-50 also emits target LLVM SHA-256
`53d2a01cc213e6974a1451ae0416c3a3bc7c79280003826f6ea8cf3def429423`, exactly the predecessor
artifact. All 53 convolutions correctly report `loop_conv_store_is_narrow_only`; none is silently
narrowed. Consequently this candidate establishes the native mechanism and selector, but deletes
no ResNet host im2col yet. The next dependency is target-neutral quantized epilogue/boundary
formation followed by a layout cost decision that can retain NHWC/HWIO. Once those facts are
present, the same selector activates without a ResNet rule.
