# Facts this backend was derived from, and where each one came from

Nothing in `mlir_oot/` is a guess.  Every hardware number, opcode, field position and semantic rule
below is quoted from a source in the granted bundle, and the module that consumes it is named.

## 1. RTL-derived hardware facts

Source: `merlin.targetgen.rtl.facts.load_facts("gemmini")["facts"]`, generator
`rtl-introspect-v2-circt-hw`, method `decoder_icmp_fanout(mlc)` over
`Gemmini_core_hw.mlir` (config `GemminiAndOPUShuttleConfig`).  Transcribed into
`mlir_oot/tables/rtl_facts.py`.

| fact | value | facts.json path | used by |
|---|---|---|---|
| mesh rows x cols | 16 x 16 | `arrays[name=mesh]` | tile loop bound `DIM`, every rows/cols field |
| scratchpad | 262144 B, depth 4096 | `memories[name=scratchpad]` | `SPAD_ROWS = 16384` operand rows; block sizing |
| accumulator | 65536 B, depth 512 | `memories[name=accumulator]` | `ACC_ROWS = 1024`; C-block sizing; verifier bound |
| operand dtype | i8 | `datapaths[name=input]` (`scratchpad smem UInt<8>`) | lane routing, mvin element width |
| accumulator dtype | i32 | `datapaths[name=accumulator]` (`AccumulatorMem SInt<32>`) | full-width readout, staging dtype |
| RoCC custom opcode | 123 (0x7b) | `interfaces[funct_decode_table].custom_opcode` | every `.insn` |
| funct3 | 3 | `interfaces[funct_decode_table].funct3` | every `.insn` (xd=0, xs1=1, xs2=1) |
| legal functs | 26 values, 0..24 + 126 | `interfaces[funct_decode_table].legal_funct` | `isa.assert_legal` |
| funct names | CONFIG_CMD, LOAD_CMD, ... | `interfaces[funct_decode_table].names` | cross-check of the header's `k_*` |

`ACC_ROWS` is computed as `65536 / (16 * 4)`, i.e. bytes / (DIM * sizeof i32), which is the
accumulator row the ISA addresses; the `depth` field in the fact bundle counts banked entries.

## 2. Instruction encoding

Source: the shipped ISA definition
`experiments/capsule_bench/targets/gemmini/contracts/hwbringup_gemmini_v0/isa_include/gemmini.h`
plus `gemmini_params.h`.  Transcribed into `mlir_oot/tables/isa.py`.

* funct7 values: the `#define k_*` block (`k_CONFIG 0`, `k_MVIN2 1`, `k_MVIN 2`, `k_MVOUT 3`,
  `k_COMPUTE_PRELOADED 4`, `k_COMPUTE_ACCUMULATE 5`, `k_PRELOAD 6`, `k_FLUSH 7`), cross-checked
  name-for-name against the RTL decode table above.
* CONFIG subtypes in `rs1[1:0]`: `CONFIG_EX 0`, `CONFIG_LD 1`, `CONFIG_ST 2`.
* `ADDR_LEN 32`, `DIM 16` — `gemmini_params.h`.
* rs1/rs2 packings, one for one from the macros:
  * `gemmini_extended_mvin` / `gemmini_extended_mvout`:
    rs1 = DRAM address, rs2 = `rows << (ADDR_LEN+16) | cols << ADDR_LEN | local`.
  * `gemmini_extended_preload`: rs1 = BD triple, rs2 = C triple, same field layout.
  * `gemmini_extended_compute_{preloaded,accumulated}`: rs1 = A triple, rs2 = BD triple.
  * `gemmini_extended3_config_ex`: rs1 `[63:32] acc_scale | [31:16] a_stride | [9] b_transpose |
    [8] a_transpose | [7] set_only_strides | [4:3] activation | [2] dataflow | [1:0] cmd_type`,
    rs2 `[63:48] c_stride | [31:0] in_shift` (the comment above the macro states this layout).
  * `gemmini_extended5_config_ld`: rs1 `[63:32] scale | [31:16] block_mvin_stride |
    [15:8] pixel_repeats | [4:3] id | [2] shrunk | [1:0] cmd_type`, rs2 = DRAM row stride.
  * `gemmini_extended2_config_st`: rs1 `[63:56] ocols | [55:48] orows | [47:40] pocols |
    [39:32] porows | [31:24] pool_out_dim | [15:10] lpad | [9:8] upad | [7:6] pool_size |
    [5:4] pool_stride | [3:2] acc_act | [1:0] cmd_type`, rs2 `[63:32] acc_scale | [31:0] stride`.
  * `gemmini_flush`: rs1 = skip, rs2 = 0.
* `WEIGHT_STATIONARY 1`, `NO_ACTIVATION 0`, `RELU 1`, `GARBAGE_ADDR 0xFFFFFFFF`.
* Local-address metadata bits from `rtl/gemmini/LocalAddr.scala` (Chisel bundle order, MSB first):
  `is_acc_addr` @31, `accumulate` @30, `read_full_acc_row` @29.  Confirmed by the worked example
  `example_kernel/matmul_ws.c`, which clears bit 30 and sets bit 29 before a full-width mvout.
* The `.insn` spelling was cross-checked against `python isa_tools.py asm`, which packs the same
  fields from this target's own derived ISA model; `disasm`/`lint` decode the emitted module back
  and every instruction reports its intended class and operands.

## 3. Weight-stationary sequence

Source: the explicit (commented) reference loop inside `sp_tiled_matmul_ws` in `gemmini.h`.  Used
for the ISA-level ordering rules only; the tiling, blocking and addressing in
`mlir_oot/lowering/schedule.py` are this compiler's own.

* `C_sp_addr_start = 3 << (ADDR_LEN-2) | full_C << (ADDR_LEN-3)` — the accumulate bit is cleared on
  the first k step so the first contribution overwrites.
* `pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR`, `COMPUTE_PRELOADED` on the first row-tile and
  `COMPUTE_ACCUMULATE` afterwards — i.e. flip the mesh weights only when a new B was preloaded.
* Ragged tiles are expressed through the `rows`/`cols` fields (`DIM - pad_*`), not by zero-filling.
* A bias is DMA'd into the accumulator before the first compute (`D_sp_addr_start = 1 << 31`) with a
  zero DRAM row stride for the repeating-vector case.

## 4. Numeric semantics

Source: `merlin/contract/command_buffer_abi.yaml`, `merlin/contract/interface_grammar.md`, and the
runtime tensor primitives in `merlin/python/merlin/runtime/tensor.py` (an allowed authoring input,
NOT imported by the package).

* `acc_scale`: `clamp_i8(round_near_even(acc * scale))` in float32 — matches the `ACC_SCALE` /
  `ROUND_NEAR_EVEN` macros in `gemmini_params.h`, so the store path performs it exactly.
* `relu`: `x if x > 0 else 0`.
* `requant`: round-half-UP arithmetic shift; the shift is required and has no default.
* `maxpool`: `Ho = (H + pt + pb - ph) // sh + 1`; `pool_padding = [top, left, bottom, right]`;
  `pool_pad_value` required whenever any pad is nonzero.
* narrowing readout to `i8`: saturate (clamp), never truncate.
* conv2d im2col column order is `(kh, kw, ci)` and out-of-bounds taps read zero
  (`merlin/python/merlin/runtime/commandbuffer.py::conv_im2col`); the same file documents the
  `params.im2col_recipes` schema this backend emits.

## 5. Kernel ABI

Source: `merlin/contract/mlir_oot_backend_contract.yaml` -> `kernel_abi`.

* `arg_order_by_command_shape` is implemented literally, rows tried top-down, in
  `mlir_oot/lowering/plan.py::kernel_args`.
* `pointee_layout: row-major, edge tiles zero-padded to a multiple of 16 (DIM)` — implemented as
  `row_pitch(cols) = ceil(cols/16) * 16`.  This was also confirmed empirically: with a tight pitch
  a 16x31 @ 31x15 capsule produced a correct-looking output whose LAST ROW was zero, which is
  exactly the signature of writing at a 60-byte pitch into a 64-byte-pitch buffer.

## 6. Facts established by measurement against the target's own oracle

These are not in any document; they were bisected with probes on the redacted self-check and are
recorded in `docs/iteration_notes.md`:

* the emitted kernel must be a SINGLE basic block (a `llvm.br`/`llvm.cond_br` loop faults);
* the emitted kernel must not use `llvm.icmp`/`llvm.select` (faults);
* the CPU lane must not write into a buffer the store DMA wrote (faults);
* the program oracle's commit-epilogue vocabulary is `relu, acc_scale, maxpool` (no bias stage);
* the program oracle builds only the RES_PACK + matmul + commit shape and the int movement shape;
  its movement path can size `i32`/`i8` destination buffers only.

## 7. Round 2 — additional documented facts, and what they are used for

* `merlin/contract/schemas/capsule.schema.json` -> `lanes`: "`require` names the execution lanes
  that must each have carried work, **using the routing plan's own keys** (e.g. `on_mesh`,
  `in_contract_vector_scalar`, `scalar_rvv_lane`) ... A named lane the routing plan does not report
  fails the capsule with that lane named."  This is the only statement of the lane vocabulary, and
  `mlir_oot/frontend/linalg_reader.py` now reports `on_mesh` / `scalar_rvv_lane` because of it.
* `merlin/contract/oracle_runner_contract.yaml` -> `execution_artifact.harness_output_format`:
  `"OUT <name> <rows> <cols> <v0> <v1> ... <v(rows*cols-1)>   # row-major integers"`.  The runner's
  readback channel is INTEGER-valued.  Combined with the measured `movement destination declares
  output dtype 'f32', which this harness has no buffer width for (sized: ['i32', 'i8'])`, this is
  why a float-result host-only module is a DELIVERY gap and declines.
* `merlin/contract/command_buffer_abi.yaml` -> `COMMIT.operands.src: acc_handle` vs the whole-op
  entries' `dst: output_tensor`.  A whole-op writes a TENSOR and `COMMIT` resolves an ACCUMULATOR,
  so `BATCHED_MATMUL -> COMMIT` cannot be chained (measured: `KeyError: 'acc_Y0'`).
* `merlin/contract/command_buffer_abi.yaml` -> `COMMIT.attributes.epilogue` and
  `merlin/runtime/commandbuffer.py::EPILOGUE_STAGES` / `BIAS_STAGES` / `bias_tensor_name` — the
  ABI's epilogue vocabulary and the rule that a bias name may ride in either the operands or the
  attributes.  The lowering writes it in both places for that reason.
* `prov.quantization` on the whole-model capsules' module (`int8_weight_only`,
  `int8_dyn_act_int8_weight`) is the only IR-derivable signal that a float contraction in a model
  module is INTENDED for the integer mesh; the host-only capsules carry no such attribute.  Not yet
  used by the placement rule (it would make the routing plan claim a lane this round cannot lower
  to) — recorded as the next round's starting point.

## Round 3 additions

* `merlin/contract/capsules/conformance/gemmini.yaml` -> `host_lane.admitted_pairs`:
  **`contraction/i8`, `elementwise_map/i8`, `movement/i8`, `reduction/i8`** — this target's
  capability manifest, stated outright.  `model/M3_host_island_seam_gemmini/capsule.pytorch.py`
  says the mesh half of it in prose: *"The two outer regions are int8 contractions because
  `contraction/int8` is the only family-and-dtype this target's capability manifest admits on its
  mesh."*  Consumed by `mlir_oot/frontend/linalg_reader.py::place`, which was already deciding it
  this way from the datapath alone; the manifest is what CONFIRMS the rule rather than changing it.
* The same loader states the numeric property the mixed-lane lowering depends on: *"every weight is
  ternary (|w| <= 1) and every activation enters a GEMM bounded by a saturating quantizer, so
  |accumulator| <= K * amp is a static property of the graph"* (96 against int8's 127).  That is why
  an i8 x i8 -> i32 mesh contraction committed at `output_dtype: i8` is exact for that capsule
  rather than a saturation approximation of torch's int8-accumulating matmul.
* `merlin/contract/schemas/command_buffer.schema.json` -> `tensors.base`: *"DRAM byte address the
  kernel loads this input from / stores this output to. The program oracle preloads inputs and reads
  the output at these addresses ... if omitted, the harness assigns a canonical layout"*, and
  `tensors.role` enum `input|weight|bias|output|scale`.  The `base` field was PROBED (round 3) and
  does not change the readback encoding; the `role` enum is why the mixed lane declares its
  compiler-owned intermediates `output` rather than `input` (see below).
* `merlin/runtime/commandbuffer.py` -> `dataflow_operands` / `_produces`: *"a declared tensor that
  some command CONSUMES and no command PRODUCES is a leaf input"*.  This is why a mixed-lane
  intermediate must not be declared `role: input` — it would be read back as one of the model's own
  inputs.  Declared `output`, `dataflow_operands` on the M3 buffer returns exactly the interface's
  five arguments and `Y0`.
* `merlin/contract/mlir_oot_backend_contract.yaml` -> `kernel_abi.arg_order_tokens`:
  `matmul_lhs_group_major` = *"each matmul's `lhs`, grouped by the resident weight its `rhs`
  resolves to"*.  Verified on a two-resident / three-matmul probe: the argument list comes out
  `[W1, W2, A0, A1, A1, Y0, Y2, Y1]` — the duplicated lhs is what the contract specifies, one
  pointer per matmul.
* xDSL behaviour, not a contract fact, but load-bearing: `IntegerAttr(0)` and an empty `ArrayAttr`
  are **falsy**, so `op.properties.get(k) or op.attributes.get(k)` silently misses a zero constant.
  `mlir_oot/codegen/host_linalg.py::attr_of` exists for that reason.

## 8. Facts added in round 4

| fact | source | used by |
|---|---|---|
| the ABI's im2col gather is `conv_im2col` / `conv_out_dims`, shared by reference + simulator + device harness, with out-of-bounds taps reading ZERO | `merlin/contract/command_buffer_abi.yaml` (`CONV2D.semantics`) and the same definition in `merlin/python/merlin/runtime/commandbuffer.py` (an allowed authoring input, NOT imported) | `lowering/plan.py:_op_matmul_batched` — the block-diagonal batched gather is expressed as an `im2col_recipe` and was checked against that definition for B = 1..5 |
| `params.im2col_recipes` is the ABI's one ADDITIVE derivation mechanism: a derived operand is materialised identically for the three engines | `commandbuffer.materialize_inputs` docstring + `command_buffer_abi.yaml` | the conv im2col matrix and the batched block-diagonal operand |
| the kernel-ABI `native_whole_op` row admits only `[ATTENTION_QK, ATTENTION_PV, CONV2D]`; anything else falls through to the resident-matmul row | `mlir_oot_backend_contract.yaml:kernel_abi.arg_order_by_command_shape` | why a lone `BATCHED_MATMUL` could not be bound, and why the batched lowering now emits the resident-matmul shape |
| `bias_add` / `bias` are ONE stage pair (`BIAS_STAGES`), and the bias name may ride in either `operands` or `attributes` | `commandbuffer.BIAS_STAGES` / `bias_tensor_name` | the commit's bias is emitted in BOTH positions; measured that the spike builder rejects either spelling |
| a fused pool's `pool_in_dims` is the conv's OWN output extent and a disagreement is "rejected, never reconciled" | `command_buffer_abi.yaml:CONV2D.attributes.pool_in_dims` | `plan.py:_op_conv2d` cross-check |
| the accumulator's residual-add idiom: move-in to `1 << (ADDR_LEN-1)`, second move-in to `3 << (ADDR_LEN-2)` (accumulate bit), move-out | `gemmini.h` `sp_tiled_resadd` + `sp_tiled_matmul_ws` (`repeating_bias`, `D_stride = 0`) | `schedule.py:bias_add` |
| the target's admitted (family, dtype) pairs are contraction/i8, elementwise_map/i8, movement/i8, reduction/i8 | `merlin/contract/capsules/conformance/gemmini.yaml:host_lane.admitted_pairs` | `frontend/linalg_reader.place` |
| `prov.weights_file` marks a WHOLE MODEL (exactly the 7 `model/` capsules carry it) | the capsule interface modules themselves | `linalg_reader.whole_model`, which decides that the runner's integer readback rule does not apply |
| the whole-model plane's engine is `merlin-compile model --target gemmini --run mesh --verify` and it does NOT read the package's routing plan | measured: the self-check's own `numeric.engine` field, plus a probe that claimed every region on the mesh and changed nothing (`iteration_notes.md` R4.3) | recorded as a scope limit in `REPORT.md` |

## 9. Facts added in round 5

| fact | source | used by |
|---|---|---|
| `ACC_SCALE(x, scale)` is `clamp(ROUND_NEAR_EVEN((float)x * scale))` — the product is taken in **f32** (`scale_t`/`acc_scale_t` are both `float`) and the clamp is to the operand dtype's range | `gemmini_params.h` lines 24–31 / 68–72 (the shipped ISA header), corroborated by `command_buffer_abi.yaml:COMMIT.attributes.acc_scale` = *"applied as `clamp_i8(round_near_even(acc * scale))`"* | `codegen/llvm_emit.py::_acc_scale` — the generated CPU-lane `acc_scale` readout, so the CPU lane and the hardware store path compute the same value |
| `ROUND_NEAR_EVEN(x)` is round-half-to-**even** (the header spells out the `rem < 0.5 / > 0.5 / i % 2 == 0` cases) | `gemmini_params.h` line 52 | `codegen/fpbuilder.py::round_near_even`, verified op-for-op against this formula on 184 024 values (0 mismatches) |
| the harness output format is `OUT <name> <rows> <cols> <v...>` **row-major integers**, and correctness is *"integer, exact `==` across all three; no FP tolerance"* | `merlin/contract/oracle_runner_contract.yaml` (`execution_artifact.harness_output_format`, `correctness`) | the stated reason on the 10 float host-only declines — a third, contract-level confirmation of the R2.3 / R3.10 measurements |
| `tensors.physical` is a physical→logical **layout** map (`{"unstack_row_halves": 2}`), and `tensors.preload_b64` is *"harness-attached input bytes; not emitted by the package"* | `merlin/contract/schemas/command_buffer.schema.json` | why neither is a float-readback or a constant-injection mechanism (both were considered and rejected as guesses) |
| the ABI's opcode vocabulary contains **no vector-map / elementwise command** — the ten opcodes are RES_PACK, MATMUL_RESIDENT, MATMUL, COMMIT, EVICT, MOVEMENT, ATTENTION_QK, ATTENTION_PV, BIAS_ADD, CONV2D | `command_buffer_abi.yaml:opcodes`, and `commandbuffer.py` defines no `VECTOR_MAP` | why a standalone post-COMMIT bias/vector add cannot be expressed, and why an admitted `elementwise_map`/`reduction` region is only reachable FUSED into a contraction's readout |
| this target expresses `elementwise_map` and `reduction` on the mesh only as epilogue stages composed with a contraction (`acc_readout_scaled -> elementwise_map composed_with contraction`) | `merlin/python/merlin/targetgen/semantic_families.py` (allowed authoring input, NOT imported) + the ISA corpus itself (`SY_elementwise_map_i8_aligned` is a matmul with a `relu` epilogue) | the decision NOT to broaden `linalg_reader.MESH_FAMILIES`, which would report a lane the kernel does not drive |
