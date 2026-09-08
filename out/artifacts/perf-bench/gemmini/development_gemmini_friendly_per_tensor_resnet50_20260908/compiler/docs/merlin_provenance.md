# Merlin provenance — `merlin_assisted_rtlchecks` (arm 4), gemmini, round 1

## 1. Merlin tools used

| Tool (path) | Used? | What you used it for |
|---|---|---|
| `merlin.targetgen.rtl.facts.load_facts("gemmini")` | **yes, ran first** | The whole hardware model. Mesh 16x16, scratchpad 262144 B, accumulator 65536 B, operand i8 / accumulator i32, RoCC custom opcode 123, funct3 3, the 26 legal functs and their decoder names. Transcribed verbatim into `mlir_oot/tables/rtl_facts.py`; nothing about the hardware was guessed. |
| `merlin.targetgen.rtl_backend.target_profile` / `derived_levers` | **yes, ran** | Returned `['spatial.dataflow', 'spatial.accumulator_resident', 'memory.capacity_fit', 'dispatch.descriptor_reuse', 'dispatch.dma_overlap', 'dispatch.loop_offloaded', 'layout.operand_major']`. Built the ones the trace gate exercises: weight-stationary dataflow (`config_ex`), accumulator-resident C blocks across the K loop, capacity-fit blocking against the two derived depths, descriptor reuse (PRELOAD with GARBAGE + `COMPUTE_AND_STAY` when B does not change), operand-major mvin. Not built: `dispatch.loop_offloaded` — the hardware loop functs (`LOOP_WS*`) would replace the MVIN/PRELOAD/COMPUTE classes the capsules' `expected.instruction_classes` require, so offloading the loop would fail the trace gate. |
| `cca_contract.check_bijection("gemmini")` | **yes, ran** | `orphan_fields=[] orphan_routes=[] clean=True` — no leverable axis was left unwired and no phantom lever to add. |
| `action_catalog.escalation_ladder(axis, "gemmini")` | **yes, ran** | `spatial.dataflow` returns one rung, `HEURISTIC` at seam `<oot_package>/lowering/`, "the generated OOT backend's command/tile-program emitter — thread the derived CodegenOpts through it". That is exactly where the dataflow decision lives in this package (`mlir_oot/lowering/schedule.py`). Other axes returned an empty ladder. |
| `targetgen/generate/` (scaffold gen) | **yes, ran** | `target_repo.generate_skeleton('gemmini')` (15 artifacts) for the package layout, and `generate.xdsl.generate(<dialect plan>)` which produced `xdsl/gemmini_dialect.py` — a real IRDL prototype with `ResidentTensorType`/`AccumulatorType` and `res_pack/matmul/commit/evict` ops plus an epilogue verifier. `mlir_oot/ir/gemmini_dialect.py` follows that generated shape (shared `_GemminiOp` parse/print, `irdl_op_definition`, `verify_` per op) with the op set this backend actually emits. `mlir_scaffold.generate(plan)` was also run; its output is C++/TableGen, which this arm forbids, so it was read for structure only. |
| `xdsl_dialects/` (dialect patterns) | **yes, read** | `interface.py` for the epilogue vocabulary, `KNOWN_OUTPUT_DTYPES`, and — importantly — the statement that the runtime's non-matmul vocabulary is only `VECTOR_MAP` (add/mul/identity + relu) and `VREDUCE` (sum). That is what ruled out expressing a bf16 softmax/layernorm region as an ABI command. |
| `targetgen/contract/interface_emit.py` | **yes, read + ran** | Ran `parse_interface_mlir` on several capsules to see the canonical command-buffer shape each interface op maps to (`RES_PACK`/`MATMUL_RESIDENT`/`COMMIT`/`EVICT`/`MOVEMENT`/`CONV2D`/`ATTENTION_QK`/`BATCHED_MATMUL`) and `defined_mnemonics()` for the closed op list. The submission does NOT use it: `mlir_oot/frontend/` defines the `merlin_iface` dialect in xDSL IRDL and parses with `xdsl.parser.Parser`. |
| `oot_starterkit/` | **read, not vendored** | Read `iface.py`, `cmdbuf.py`, `verify.py`, `transforms.py` for the intended shape of the four seams. The submission implements its own equivalents so the package stays self-contained. |
| `runtime/commandbuffer.py`, `runtime/tensor.py` | **read** | The single definitions of `conv_im2col` (column order, zero OOB taps), the `params.im2col_recipes` schema, `maxpool2d_rows`, `requant_acc_scale` (round-half-even f32) and `to_i8` (saturate). These fixed the numeric semantics the backend has to reproduce. Not imported. |
| `isa_tools.py` (asm / disasm / lint / debug) | **yes, ran throughout** | `asm` confirmed the canonical `.insn r 0x7b, 0x3, <funct>, x0, $0, $1` spelling and the `llvm.mlir.constant(<v> : i64)` operand form before any encoder code was written. `lint`+`disasm` on every emitted artifact: 0 UNKNOWN everywhere, and the decoded operand fields (DRAM arg index + offset, row pitch, readout dtype, accumulate bit, acc_scale, relu bit) were reconciled against the command buffer. `debug` was run on `A6_resident_reuse` for per-command hardware state. |

## 2. Files generated with Merlin tooling

| submission file | origin | notes |
|---|---|---|
| `mlir_oot/ir/gemmini_dialect.py` | mixed | Structure from `generate.xdsl.generate` (IRDL op class shape, shared parse/print, per-op `verify_`); the op set, the RTL-capacity verifiers and the host-lane ops are this backend's. |
| `mlir_oot/tables/rtl_facts.py` | generated-from-facts | A transcription of `load_facts("gemmini")["facts"]`. |
| `mlir_oot/tables/isa.py` | hand, from the shipped ISA | Packings copied field-for-field from the `gemmini.h` macros; cross-checked against `isa_tools asm`. |
| package layout (`frontend/ ir/ lowering/ codegen/ tables/`) | mixed | Follows `target_repo.generate_skeleton` minus the C++ directories this arm forbids. |
| everything else | hand | Front end, plan, tile schedule, LLVM emitter, command buffer. |

## 3. Failures encountered, and which Merlin tooling diagnosed them

| round | capsule | failure plane / violations | fix | Merlin tool that helped |
|---|---|---|---|---|
| 1 | `SY_contraction_i8_partial` | `spike` functional_mismatch; last output row zero | DRAM row pitch padded to a whole tile (`ceil(cols/16)*16`) | none — read from the emitted `OUT` line and the kernel-ABI `pointee_layout` note |
| 1 | `SY_contraction_i8_partial` | `trace does not open with a FENCE` | emit `fence` before `flush` | the self-check's own trace_check |
| 1 | `A5_relu_epilogue` | `mode relu declared but no CONFIG_ST sets relu activation` | CONFIG_ST always encodes the declared activation | the self-check's trace_check |
| 1 | every CPU-lane capsule | `spike` fault, empty console | single-block, select-free kernel | bisected with probes; `isa_tools lint/disasm` confirmed the ISA half was clean, which is what localised it to the CPU lane |
| 1 | `A5`, `SY_epilogue_relu`, `SY_elementwise_map_i8_sub_tile` | `spike` fault at 313 emitted lines | stop writing into the DMA-written staging buffer; fold the stages into the gather | `lint` again ruled out the encoding |
| 1 | `GP0_matmul_maxpool_i8` | `mode i8 declared but no MVOUT has i8 readout` | readout plan: narrow the store path first, pool on the CPU lane | the self-check's trace_check |
| 1 | `SY_epilogue_bias_add` | `unsupported epilogue stage 'bias_add'` | none — the target's program oracle implements `relu/acc_scale/maxpool` only | the oracle's own message |
| 1 | 10 float host-lane capsules | `backend_declined` -> `lanes: on_mesh forbidden` -> `movement destination declares bf16, no buffer width` | emit ZERO accelerator instructions for a host-placed region (the `lanes` gate now passes); the float CPU program still has no command shape to ride on | `xdsl_dialects/interface.py` explained why no ABI opcode expresses these regions |

## 4. Files changed per iteration

| round | files changed | result |
|---|---|---|
| 1 (build-out) | whole package created | 0 -> 80/94 |
| 1 (row pitch) | `lowering/plan.py`, `lowering/schedule.py` | fixed every ragged shape |
| 1 (kernel form) | `codegen/builder.py`, `codegen/llvm_emit.py` | fixed every CPU-lane capsule |
| 1 (readout plan) | `lowering/schedule.py`, `codegen/llvm_emit.py` | fixed the i8-readout trace violation |
| 1 (host lane) | `frontend/linalg_reader.py`, `lowering/host_lane.py` | cleared the `lanes` protocol violation; the capsules still decline |

## 5. Final-artifact integrity (self-attestation)

- Imports Merlin runtime code? **no** — `grep -rn "import merlin\|from merlin\|reference_outputs\|pipeline.execute\|runtime.reference\|runtime.simulator" submission/` returns nothing.
- Self-contained (graded only through its CLI entrypoints)? **yes** — the package imports `xdsl` and the standard library only.
- Merlin authoring artifacts accidentally in `submission/`? **none** — the generated scaffolds were written to `/tmp/scaf` and read, not copied. No `runtime_adapter`, no C++/TableGen, no `build:` block.
- No regular expressions anywhere in the package (checked); both input grammars are parsed with `xdsl.parser.Parser`.

## 6. One-line summary

The RTL fact bundle and `isa_tools` were decisive — the mesh/capacity numbers and the exact `.insn`
spelling came from them rather than from guesswork, and `lint`/`disasm` kept ruling the encoding
out so every remaining bug could be localised elsewhere; the xDSL scaffold saved the dialect
boilerplate; the CCA ladder confirmed the one seam worth building. What Merlin tooling did NOT help
with was the three behaviours that cost the most time (padded DRAM row pitch, single-block
select-free kernels, write-once staging buffers) — those are properties of the program oracle that
only bisecting against the redacted self-check revealed.


---

# Round 2 addendum

## Merlin tools invoked this round (before the first submission edit)

| call | result actually returned |
|---|---|
| `merlin.kernels.cca_contract.check_bijection('gemmini')` | `BijectionReport(orphan_fields=[], orphan_routes=[], unclassified=[], ladder_errors=[])` — the lever set is complete; nothing left to wire and no phantom route. |
| `merlin.kernels.action_catalog.escalation_ladder(axis, 'gemmini')` for `spatial.dataflow`, `dispatch.loop_offloaded`, `memory.capacity_fit` | each returns a single rung whose `seam_file` is `<oot_package>/lowering/` — i.e. every remaining lever is applied in this package's own tile-program emitter, which is where round 2's work went. |
| `merlin.targetgen.rtl_backend.derived_levers(target_profile('gemmini'))` | `['spatial.dataflow', 'spatial.accumulator_resident', 'memory.capacity_fit', 'dispatch.descriptor_reuse', 'dispatch.dma_overlap', 'dispatch.loop_offloaded', 'layout.operand_major']` |
| `merlin.targetgen.rtl.facts.load_facts('gemmini')` | mesh 16x16, scratchpad 262144 B / depth 4096, accumulator 65536 B / depth 512, input datapath i8, accumulator datapath i32 — re-confirmed against `mlir_oot/tables/rtl_facts.py`. |
| `merlin.targetgen.generate.target_repo.generate_skeleton('gemmini')` | 15 relpaths (`README.md`, `AGENT.md`, `pyproject.toml`, `docs/`, `contracts/`, `xdsl/`, `include/`, `lib/`, `tools/`, `runtime/`, `zephyr/`, `llvm/`, `examples/`, `tests/`) — re-run as the round's generation witness; the package layout already follows it. |

## What Merlin tooling did and did not help with in round 2

* **Helped.** `merlin/contract/schemas/capsule.schema.json`'s `lanes` description is the only place
  the ROUTING PLAN's lane keys are named (`on_mesh`, `scalar_rvv_lane`); the router now reports
  those instead of its own `mesh`/`host` spellings. `merlin/contract/command_buffer_abi.yaml` and
  `merlin/runtime/commandbuffer.py` (an allowed authoring module) are what established that
  `BATCHED_MATMUL` writes a TENSOR while `COMMIT` resolves an ACCUMULATOR, which is why the two
  cannot be chained — that saved a round of guessing. `oracle_runner_contract.yaml`'s one-line
  `harness_output_format` (`# row-major integers`) independently confirmed the float-readback wall
  that the probe in `iteration_notes.md` R2.3 measured.
* **Did not help.** Nothing in the granted surface states this target's *capability manifest*, so
  `M3_host_island_seam_gemmini`'s `FALLBACK_ON_ELIGIBLE_REGION` (21 admitted regions) still cannot
  be answered from the contract; and neither `check_bijection` nor `derived_levers` says anything
  about the program oracle's command-shape whitelist, which is what actually blocks the batched and
  bias_add capsules. Both of those were found by measurement against the redacted self-check.

## Files changed in round 2

| file | origin | change |
|---|---|---|
| `mlir_oot/codegen/fpbuilder.py` | hand (new) | branch-free f32 scalar set; generated exp/erf/rsqrt/log/powf/tanh; bf16 bit format |
| `mlir_oot/codegen/host_linalg.py` | hand (new) | structural `linalg-on-tensors` -> CPU-lane lowering |
| `mlir_oot/codegen/llvm_emit.py` | hand | `host_linalg` instruction kind |
| `mlir_oot/codegen/gemmini_module.py` | hand | `gemmini.host_lane_program` op; opaque-attr filter; lane keys |
| `mlir_oot/ir/gemmini_dialect.py` | hand | `HostLaneProgramOp` with a verifier |
| `mlir_oot/frontend/linalg_reader.py` | hand | lane vocabulary from `capsule.schema.json` |
| `mlir_oot/lowering/host_lane.py` | hand | carrier command shape, lane report, two distinct declines |
| `mlir_oot/lowering/plan.py` | hand | batched matmul as the ABI whole-op; bias split investigated and reverted |
| `manifest.yaml` | hand | `components` widened to the paths the command-buffer path now rides on |

## Final-artifact integrity (unchanged)

- Imports Merlin runtime code? **no**.
- Self-contained (graded only through the CLI entrypoints)? **yes**.


---

# Round 3 addendum

## Merlin tools invoked this round (before the first submission edit, results non-empty)

| call | result actually returned |
|---|---|
| `merlin.kernels.cca_contract.check_bijection('gemmini')` | `BijectionReport(backend='gemmini', orphan_fields=[], orphan_routes=[], unclassified=[], ladder_errors=[])` — still no leverable axis unwired and no phantom route. |
| `merlin.kernels.action_catalog.escalation_ladder('spatial.dataflow','gemmini')` | one rung: `{'action_class': 'HEURISTIC', 'target_seam': 'rtl_codegen:spatial.dataflow', 'seam_file': "<oot_package>/lowering/", 'forkable_now': False, 'needs_new_code': True}` — and `lowering/` is exactly where this round's work went (`lowering/model_lane.py`). |
| `merlin.targetgen.rtl_backend.target_profile('gemmini')` / `derived_levers(...)` | `TargetProfile(legal_opcodes=(0..24,126), dim=16, memory_map={scratchpad 262144 B depth 4096, accumulator 65536 B depth 512})`; levers `['spatial.dataflow', 'spatial.accumulator_resident', 'memory.capacity_fit', 'dispatch.descriptor_reuse', 'dispatch.dma_overlap', 'dispatch.loop_offloaded', 'layout.operand_major']`. |
| `merlin.targetgen.rtl.facts.load_facts('gemmini')` | `funct_decode_table` with `custom_opcode 123`, `funct3 3`, `legal_funct [0..24, 126]` and the decoder NAMES (`CONFIG_CMD`, `LOAD_CMD`, `STORE_CMD`, `COMPUTE_AND_FLIP_CMD`, ...), plus the `ReservationStation` / `FrontendTLB` interfaces — re-checked against `mlir_oot/tables/rtl_facts.py`, unchanged. |
| `merlin.targetgen.generate.target_repo.generate_skeleton('gemmini')` | 15 relpaths (`README.md`, `AGENT.md`, `pyproject.toml`, `CMakeLists.txt`, `docs/`, `contracts/`, `xdsl/`, `include/`, `lib/`, `tools/`, `runtime/`, `zephyr/`, `llvm/`, `examples/`, `tests/`) — the round's generation witness. |

## What Merlin tooling DID help with this round

* **The capability manifest, at last.** `merlin/contract/capsules/conformance/gemmini.yaml`
  (`host_lane.admitted_pairs`) states this target's admitted (family, dtype) pairs outright —
  `contraction/i8`, `elementwise_map/i8`, `movement/i8`, `reduction/i8` — and the M3 capsule's own
  loader says the same in prose. That closed round 1's open question (`FALLBACK_ON_ELIGIBLE_REGION`
  is not a placement-rule bug) and is what made the mixed-lane pass worth building.
* **`merlin/runtime/commandbuffer.py`** (an allowed authoring module) supplied `dataflow_operands`,
  which is how the mixed-lane buffer's intermediate tensors were checked: declared `role: input`
  they showed up as MODEL INPUTS in the dataflow, declared `role: output` the leaves come back as
  exactly the interface's own five arguments.
* **`merlin/contract/schemas/command_buffer.schema.json`** documents `tensors.base` ("the program
  oracle preloads inputs and reads the output at these addresses"), which is what made the second
  float-readback probe worth running. It refuted the idea rather than confirming it — see
  `iteration_notes.md` R3.10 — but it is exactly the kind of thing that is cheaper to refute than
  to keep wondering about.
* **`isa_tools lint`/`disasm`** on every emitted artifact again: 0 UNKNOWN, and the decoded MVIN /
  MVOUT operands come back as `{'kind': 'argbase', 'arg_index': N, 'offset': 0}` — no baked DRAM
  address anywhere.

## What it did NOT help with

`qa/tier_state.json` — not a Merlin tool, just the QA gate's own bookkeeping — was what revealed the
round's most consequential defect: `lower_interface_to_target` had been returning the EMPTY STRING
for `M2_microvit_gemmini` and `SY_micro_model` (digest `e3b0c442…`, the SHA-256 of nothing). Nothing
in the granted Merlin surface says what the model plane requires of an OOT package, and the
`merlin-compile model` engine's region model (which reports "21 admitted regions" for a capsule whose
IR has 8 provenance regions) is still not derivable from anything granted.

## Files changed in round 3

| file | origin | change |
|---|---|---|
| `mlir_oot/lowering/model_lane.py` | hand (new) | the mixed-lane segmenter: mesh contractions + host runs + the DRAM buffers between them |
| `mlir_oot/codegen/host_linalg.py` | hand | integer domain (i64 SSA, modular in the declared width), `linalg.index`, `tensor.extract` with a compile-time index, `estimate_cost`, `attr_of` (the falsy-attribute fix), `run_segment` |
| `mlir_oot/codegen/fpbuilder.py` | hand | `sitofp` / `fptosi` (the latter built from the IEEE fields, since this LLVM dialect has no fptosi) |
| `mlir_oot/codegen/builder.py` | hand | integer div/rem/xor, variable-amount shifts, `wrap_int` |
| `mlir_oot/codegen/llvm_emit.py` | hand | `host_segment` emission through one shared `HostLinalg`; `declined_artifact` |
| `mlir_oot/codegen/gemmini_module.py` | hand | `declined_module`; the `host_segment` op mapping |
| `mlir_oot/lowering/host_lane.py` | hand | integer element types admitted; the up-front cost refusal |
| `mlir_oot/lowering/plan.py` | hand | `COMMIT` carries its bias as an ABI OPERAND as well as an attribute |
| `mlir_oot/lowering/schedule.py` | hand | schedules a `HostSegment` |
| `mlir_oot/gemmini_opt.py` | hand | mixed-lane routing; artifact proven during lowering; a decline answered in each entrypoint's own language |
| `manifest.yaml` | hand | `components` widened: the lowering commands now ride on all of `mlir_oot/codegen/` |

## Final-artifact integrity (re-checked this round)

- `import merlin` / `from merlin` / `reference_outputs` / `pipeline.execute` / `runtime.reference` /
  `runtime.simulator` in `submission/`: **none** (grep + an AST import scan over every `.py`).
- `import re` or any regex use: **none** (AST import scan).
- Self-contained: the package imports `xdsl` and the standard library only.

## Late-round addition (round 3): the certificate leak

Not a Merlin tool, but worth recording next to the tool notes because it is what the QA gate's own
bookkeeping revealed: `.qa_channel/broker.log` showed every capsule's certificate being
`invalidated by <unattributed> (changed)` on every round. The unattributed path was
`submission/mlir_oot/__pycache__/` — the SUBDIRECTORY caches are covered by the `mlir_oot/<subdir>/`
component entries, the top-level one was covered by nothing. The CLI shim now sets
`sys.dont_write_bytecode = True` so no bytecode is written at all, and `components:` attributes the
directory to all four commands regardless. 31 files under `submission/`, 0 unattributed.

---

# Round 4 addendum

## Merlin tools invoked this round (before the first submission edit, all results non-empty)

| call | result |
|---|---|
| `merlin.kernels.cca_contract.check_bijection('gemmini')` | `BijectionReport(orphan_fields=[], orphan_routes=[], unclassified=[], ladder_errors=[])` — the lever set is complete; nothing to wire and no phantom to remove |
| `merlin.kernels.action_catalog.escalation_ladder('spatial.dataflow','gemmini')` | one `HEURISTIC` row, seam `<oot_package>/lowering/`, `forkable_now: False`, `needs_new_code: True` — which is where this round's two scheduling changes (`bias_add`, `movement`) landed |
| `merlin.targetgen.rtl_backend.derived_levers(target_profile('gemmini'))` | the same seven axes; profile `dim=16`, scratchpad 262144 B, accumulator 65536 B, 26 legal opcodes |
| `merlin.targetgen.rtl.facts.load_facts('gemmini')` | the funct decode table with names (`0 CONFIG_CMD`, `1 LOAD2_CMD`, `2 LOAD_CMD`, `3 STORE_CMD`, `4 COMPUTE_AND_FLIP_CMD`, ... `126`) |
| `merlin.targetgen.generate.target_repo.generate_skeleton('gemmini')` | 15 scaffold paths (`README.md`, `AGENT.md`, `pyproject.toml`, `xdsl/`, `lib/`, `tools/`, `runtime/`, `tests/`, ...) |

Two further authoring reads mattered more than usual this round, and both are on the granted list:

* **`merlin.runtime.commandbuffer`** (ALLOWED authoring tool; NOT imported by the package). Its
  `conv_im2col` / `conv_out_dims` / `materialize_inputs` are the ABI's single definition of the one
  additive operand-derivation mechanism. Reading them is what turned "a batched matmul cannot be
  expressed in the resident-matmul command shape" into "a batched matmul is one contraction over a
  block-diagonal operand, and the block-diagonal operand is an im2col recipe". They were then used
  as an oracle-free CHECKER: the derived gather was run through `conv_im2col` itself for
  `B` = 1, 2, 3, 4, 5 and its product compared against the batched reference computed in plain
  Python. Both exact. No golden, no simulator, one second.
* **`merlin.targetgen.semantic_families.from_prov`** — the prov-family to semantic-family map
  (`cast/dtype_cast -> elementwise_map`, `normalization/layer_norm -> normalization`, ...), used to
  read `conformance/gemmini.yaml:host_lane.admitted_pairs` against the model capsules' own region
  tags while diagnosing the model plane.

## What Merlin tooling DID help with this round

1. **The batched contraction (+2 capsules, both certified at L3).** Above.
2. **Diagnosing the model plane, conclusively.** The self-check's own reply carries
   `numeric.engine = "merlin-compile model --target gemmini --run mesh --verify"` and
   `numeric.measured_on = "host_lane_fallback"`. Together with a probe that claimed every M3 region
   on the mesh and changed the verdict by nothing, that establishes the whole-model plane compiles
   with its own model compiler and does not read this package's routing plan. Recorded rather than
   guessed at for a fourth round.
3. **`isa_tools lint` / `disasm`** on every emitted artifact and every probe: 0 UNKNOWN, every DMA
   address an `argbase`, and the decoded class histogram is what proved the new `bias_add` emits the
   six classes its own capsule requires.

## What it did NOT help with

The two remaining oracle-side walls (`bias_add`'s stage-name rejection and the float readback
encoding) are properties of the target's program oracle, which no granted tool describes; both were
re-measured this round through the redacted self-check and both are unchanged. The whole-model
plane's invocation interface is likewise not derivable from any granted document.

## Files changed in round 4

| file | how | what |
|---|---|---|
| `mlir_oot/lowering/plan.py` | hand, from `commandbuffer.conv_im2col` | batched matmul -> block-diagonal im2col + one contraction; the conv `pool_in_dims` cross-check; the pool row-divisibility check |
| `mlir_oot/lowering/schedule.py` | hand, from `gemmini.h` `sp_tiled_resadd` | `bias_add` folded into the accumulator read-out; the movement container-legality rule |
| `mlir_oot/frontend/linalg_reader.py` | hand | `whole_model` derived from `prov.weights_file` |
| `mlir_oot/lowering/host_lane.py` | hand | a whole model is not refused on readback grounds |
| `docs/` | hand | this file, `PLAN.md` §10, `iteration_notes.md` R4.x, `public_facts_used.md` §8 |

## Final-artifact integrity (re-checked this round)

- `import merlin` / `from merlin` / `reference_outputs` / `pipeline.execute` / `runtime.reference` /
  `runtime.simulator` anywhere under `submission/`: **none**.
- `import re` or any regex use: **none**.
- Self-contained: the package imports `xdsl` and the standard library only.

## Round 5

**Merlin tools actually invoked (all before the first submission edit, all non-empty):**

| call | returned |
|---|---|
| `merlin.kernels.cca_contract.check_bijection('gemmini')` | `orphan_fields=[] orphan_routes=[] unclassified=[] ladder_errors=[]` — the lever set is complete, nothing left to wire and no phantom |
| `merlin.kernels.action_catalog.escalation_ladder('spatial.dataflow','gemmini')` | one `HEURISTIC` row, seam `<oot_package>/lowering/`, `forkable_now: False` |
| `merlin.targetgen.rtl_backend.derived_levers(target_profile('gemmini'))` | the same seven axes (`spatial.dataflow`, `spatial.accumulator_resident`, `memory.capacity_fit`, `dispatch.descriptor_reuse`, `dispatch.dma_overlap`, `dispatch.loop_offloaded`, `layout.operand_major`) |
| `merlin.targetgen.rtl.facts.load_facts('gemmini')` | the funct decode table — custom opcode 123, funct3 3, the 26 legal functs — plus the ReservationStation / FrontendTLB module evidence |
| `merlin.targetgen.generate.target_repo.generate_skeleton('gemmini')` | 15 scaffold paths |

**What Merlin tooling actually helped with this round.** Less than in earlier rounds, and honestly
so: the lever set has been complete since round 2, so `check_bijection` and `derived_levers`
confirmed rather than directed. The two tools that did real work were
`merlin.targetgen.semantic_families` (read, never imported), whose `_PROV_FAMILY` /
`ISA_ROLE_FAMILY` tables state that this target reaches `elementwise_map` only *composed with*
`contraction` — that is what stopped a plausible-looking broadening of the lane router that would
have reported a lane the kernel does not drive — and `isa_tools disasm`, whose decoded operand
fields were read back against the command buffer's own declarations (§R5.7) and matched on every
one.

**The round's two real defects were found by neither.** Both came from probing attribute
combinations the corpus does not contain: an `acc_scale` epilogue outside the single readout shape
the public capsules use, and a codegen-only decline that made the four entrypoints contradict each
other. The ISA header (`gemmini_params.h`, a shipped hardware spec rather than a Merlin tool) is
what made the fix derivable instead of guessed — `ACC_SCALE` and `ROUND_NEAR_EVEN` are written out
there, and the generated code was checked op-for-op against them.

## Files changed in round 5

| file | how | what |
|---|---|---|
| `mlir_oot/codegen/fpbuilder.py` | hand, from `gemmini_params.h:ROUND_NEAR_EVEN` | `round_near_even` — branch-free round-half-to-even, verified against the header's formula on 184 024 values |
| `mlir_oot/codegen/llvm_emit.py` | hand, from `gemmini_params.h:ACC_SCALE` + `command_buffer_abi.yaml` | `_acc_scale` — the generated CPU-lane `acc_scale` readout, so a non-narrowing or negative-scale commit lowers instead of refusing |
| `mlir_oot/lowering/schedule.py` | hand | `_check_host_stages` — an unimplemented readout stage is refused at PLAN time, so every entrypoint answers the same way |
| `mlir_oot/gemmini_opt.py` | hand | the codegen backstop restates the decline on the command buffer and prints a module, instead of exiting 1 with empty stdout |
| `docs/` | hand | `iteration_notes.md` R5.x, `public_facts_used.md` §9, `REPORT.md` §Round 5, this section |

## Final-artifact integrity (re-checked in round 5)

- AST scan over all 24 `.py` files under `submission/`: no `import merlin` / `from merlin`, no
  `import re`, no reference/simulator/oracle import. **0 violations.**
- 31 files under `submission/`, **all 31 attributed** by `components:`; no `__pycache__`, no `.pyc`.
- `isa_tools lint` over all 87 emitted artifacts: 74 845 instructions, **0 UNKNOWN**; `disasm`:
  **0 baked DRAM addresses**.
