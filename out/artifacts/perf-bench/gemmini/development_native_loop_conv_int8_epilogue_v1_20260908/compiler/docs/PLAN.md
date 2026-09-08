# PLAN.md — gemmini MLIR OOT target backend (xDSL / Python)

Written round 1, before any code. Refined, not replaced, on later rounds.

## 1. Corpus (discovered, 111 public capsule dirs under isa/ layers/ model/ model_slices/)

Two input grammars:
* **`merlin_iface` v0.1** (83 capsules) — integer, `compare: exact_int`.
  Ops seen: `tensor`, `resident_pack`, `matmul`, `commit`, `evict`, `movement`, `conv2d`,
  `bias_add`, `attention_qk`. Epilogues: `[]`, `relu`, `acc_scale`, `bias_add`, `requant`,
  `maxpool`. Output dtypes i32 / i8. Shapes from 15x15 tails to multi-tile M/K/N, deep-K
  spills, conv k3..k16 with stride/pad/dilation.
* **`linalg-on-tensors`** (28 capsules) — bf16/f32/model, `compare: tolerance_float`.
  Ops: softmax, gelu, geglu, layernorm, add, bias_add, maxpool2d, avgpool2d, global_average,
  depthwise_conv2d, reduce_sum, attention_full, fused_matmul_bias, plus 7 whole-model capstones.
  9 of these are explicitly `semantic.eligible: false` / `lanes.forbid: [on_mesh]` (host lane).

Distinct cases to cover: (op) x (dtype i8/i32 vs bf16/f32) x (epilogue set) x (tile regime:
sub-tile / partial / aligned / multi-tile in each of M,K,N / spill).

## 2. Input ingestion (`parse`)
Structural, no regex, no hand lexer:
* Define the `merlin_iface` **input dialect** as real xDSL IRDL ops/types (tensor, resident_pack,
  matmul, commit, evict, movement, conv2d, bias_add, attention_qk; `!merlin_iface.resident`,
  `!merlin_iface.acc<T>`), load into an xDSL `Context`, parse with `xdsl.parser.Parser`, then
  `module.verify()`. Broken graph ⇒ nonzero exit with a diagnostic.
* Route on the module: a module carrying `merlin_iface.version` is grammar A; otherwise parse
  with xDSL's builtin/func/linalg/tensor/arith/math dialects (grammar B) and walk the real IR.
* Reject an unimplemented `merlin_iface.version`.

## 3. Target dialect + lowering
`gemmini` xDSL dialect (IRDL, with verifiers):
types `!gemmini.spad_tile`, `!gemmini.acc_tile`; ops `gemmini.flush`, `gemmini.config_ex`,
`gemmini.config_ld`, `gemmini.config_st`, `gemmini.mvin`, `gemmini.preload`,
`gemmini.compute_preloaded`, `gemmini.compute_accumulated`, `gemmini.mvout`, `gemmini.fence`.
Verifiers check: DIM-bounded rows/cols, scratchpad/accumulator addresses inside the
RTL-derived capacities (262144 B operand / 65536 B accumulator), legal funct only.

Passes (xDSL `ModulePass` / pattern rewrites), in order:
1. `ConvertIfaceToGemmini` — semantic normalization: conv2d ⇒ im2col matmul plan; movement ⇒
   load/store plan; attention_qk ⇒ transposed-rhs matmul; bias_add ⇒ accumulator-domain add.
   Produces a target-independent **tile program** (`M,K,N` loop nest over DIM=16 tiles).
2. `ScheduleTiles` — assign scratchpad rows for A and B tiles, accumulator rows for C tiles,
   honouring the derived capacities; emit the gemmini-dialect op sequence.
3. `EmitCommandBuffer` — the same normalized plan serialized to ABI opcodes.
4. `EmitTargetArtifact` — gemmini dialect ⇒ `llvm.func @gemmini_kernel` of `llvm.inline_asm`
   `.insn r 0x7b, 0x3, <funct>, x0, $0, $1` with every operand an SSA `llvm.mlir.constant` or a
   `llvm.ptrtoint` of a kernel pointer argument.

## 4. Encoding
Derived, never invented:
* RoCC: custom opcode **0x7b**, funct3 **0x3** (xd=0,xs1=1,xs2=1), funct7 = the k_* funct from
  `gemmini.h` — cross-checked against the RTL decode table in `rtl.facts.load_facts('gemmini')`
  (26 legal functs, names 0..24 + 126).
* rs1/rs2 packing taken verbatim from the `gemmini.h` macros (mvin/mvout: rs1=DRAM addr,
  rs2 = rows<<48 | cols<<32 | spad; preload/compute: rows<<48|cols<<32|addr in both;
  config_ex/ld/st: the documented bitfields with the subtype in rs1[1:0]).
* Check before grading: `python isa_tools.py asm` on the same listing and diff against my own
  encoder; then `lint` + `disasm` on the emitted `.mlir`.

## 5. Addressing + termination
* Every DRAM address = `llvm.ptrtoint` of the kernel pointer argument for that tensor, plus a
  constant byte offset. Never a literal DRAM base.
* Argument order from `mlir_oot_backend_contract.yaml.kernel_abi.arg_order_by_command_shape`
  (movement / native_whole_op / resident_matmul), matched top-down.
* Scratchpad/accumulator addresses are fixed constants (accumulator tagged with bit 31).
* Termination: trailing `gemmini_fence` (`fence`) then `llvm.return`; FLUSH first.

## 6. Verification loop (cheapest first)
1. `python -m mlir_oot.gemmini_opt` on one capsule — parse / lower / cmdbuf / artifact (ms).
2. `isa_tools.py lint` + `disasm` on the emitted `.mlir`; reconcile decoded fields against the
   command buffer (address, stride, dtype, accumulate bit).
3. `agent_selfcheck.py --capsules <one>` for the capsule just changed.
4. `agent_selfcheck.py --shape-coverage` after any change to the tile loop.
5. `agent_selfcheck.py --capsules all` only before declaring the round done.
6. Read `qa/verdict.json` (incl. `rtl_checks`) each round and fix by `failure_plane`.

## 7. Declines
Anything the lowering cannot express (float/host-lane families with no integer datapath, an
unknown op, an epilogue stage with no derived encoding) emits `declined: {reason, op, shape}`
and **no** commands — never an empty program.

## 8. Refinement (round 2) — the host lane is a compiled lane, not a refusal

Round 1 answered an all-host module with an empty kernel plus a decline. That is only half the
plan: §7's "declines" is for what the compiler cannot LOWER, and a float elementwise/reduction/
normalization region is something it can. So the plan now has a third lowering path beside the
tile schedule and the command-buffer serializer:

* **`codegen/fpbuilder.py`** — the scalar instruction set of the CPU lane: branch-free,
  select-free f32 (per §3's single-block constraint), with `exp`/`erf`/`rsqrt` GENERATED as
  range-reduced polynomial sequences rather than called into a math library the bare-metal
  harness does not link. bf16 is a storage format handled with explicit bit arithmetic.
* **`codegen/host_linalg.py`** — a structural lowering of the `linalg-on-tensors` grammar onto
  those primitives, driven by each op's own attributes (`indexing_maps`, `iterator_types`,
  `dimensions`, `permutation`, `static_offsets/sizes/strides`) and by result types. It is a
  general evaluator over the grammar, not a table of recognized patterns.
* **`lowering/host_lane.py`** decides the COMMAND SHAPE the generated program is handed over in,
  and reports the routing plan in the contract's own lane vocabulary (`on_mesh` /
  `scalar_rvv_lane`, from `capsule.schema.json:lanes`).

The decline rule of §7 is correspondingly narrowed and made specific, because the two gaps are
different failures and must read differently in the verdict:
* a region whose element type the CPU lane has no scalar format for -> LOWERING gap;
* a module whose whole result is float and which places nothing on the mesh -> DELIVERY gap:
  this target's runner contract prints `OUT ... # row-major integers` and its movement path
  sizes a destination at i32/i8 only, so there is no readback encoding for the value (measured;
  see `docs/iteration_notes.md` R2.3). The program is still generated and emitted.

## 9. Round-3 refinement — the mixed lane (added when the strategy changed)
A `linalg-on-tensors` module is no longer all-mesh or all-host. `lowering/model_lane.py` splits the
entry function into ORDERED SEGMENTS — a mesh contraction, or a run of host ops — and gives every
value that crosses a segment boundary a DRAM buffer:
* eligibility = the placement rule (`frontend/linalg_reader.place`: family + operand dtype against the
  RTL-derived datapath) AND expressibility (a rank-2 `linalg.matmul` at the mesh operand dtype whose
  `outs` is a zero fill);
* an interface tensor a host segment touches is made resident, which is what puts its pointer in the
  kernel argument list under the ABI's resident-matmul row;
* the CPU lane carries integer tensors in the INTEGER domain (i64 SSA, modular in the declared width)
  and float tensors in f32, bridged only where the IR bridges them (`arith.sitofp` / `arith.fptosi`);
* every entrypoint must answer the same way about the same capsule, so the artifact is EMITTED during
  lowering and a codegen-only refusal becomes a stated decline rather than an empty stdout.

## 10. Round-4 refinement — three additions, all derived from documents already in the plan

* **A rank-3 contraction is a rank-2 one.** §3's normalization list gains a batched case:
  `matmul_batched` lowers to ONE `[B*M, B*K] x [B*K, N]` contraction over the BLOCK-DIAGONAL
  activation, which is derived by the ABI's own `params.im2col_recipes` gather (window `kh = B`,
  `dilation = -B`, `stride = B + 1`; an out-of-bounds tap reads zero, and those zeros ARE the
  off-diagonal blocks). No per-batch slice tensor and no new opcode -- the same
  `RES_PACK / MATMUL_RESIDENT / COMMIT` shape §5's argument order already binds. The expansion is
  quadratic in the batch, so it is budgeted and DECLINED past it per §7.
* **The standalone `bias_add` belongs on the accumulator, not the CPU lane.** §3 said
  "bias_add => accumulator-domain add" and the scheduler was doing it on the host; it now emits the
  ISA header's own idiom (a repeating move-in with a zero DRAM row stride, a second move-in with the
  accumulate bit, a full-width move-out).
* **A whole model is not delivered through the integer readback.** §8's DELIVERY-gap rule is about
  the runner's `OUT ... # row-major integers` protocol. A module that names its parameter file
  (`prov.weights_file`) is graded by the model engine instead, so the rule does not apply to it and
  the generated program is handed over rather than refused.

The verification loop of §6 gains one step, ahead of everything else: a derived operand
(an im2col matrix, a block-diagonal gather) is checked against the ABI's OWN definition of the
derivation for a spread of extents BEFORE any simulator sees it. That is oracle-free and it is what
proved the batched lowering at B = 1, 2, 3, 4, 5 in one second.
