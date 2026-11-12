# src/merlin/compiler/kernel_ir

## Purpose

The schedule-level compiler core: numerics contracts, kernel schedules and the gates that check them.
It is the action space the performance loop optimizes over. The loop edits schedules and planner
decisions, never compiler source. It also gives the correctness side a representation that is
checkable before any simulator runs.

## Subpackages

- `contract/` — numerics contracts. A contract is the declared arithmetic a golden is computed under:
  scale granularity, rounding, saturation and activation. It is built from a target's derived readout
  facts.
- `check/` — simulator-free correctness gates: exhaustive epilogue enumeration (`epilogue_enum`) and
  G0 static legality of a kernel against an instruction set (`static`).
- `ir/` — the kernel IR `mk`: loops over one target's schedule instructions; canonical text + digest.
- `isa/` — the shape of a target's schedule instruction set (`InstrDef`, `InstructionSet`); a target
  provides one through its backend hook `sched_instruction_set()`.
- `codegen/` — emit a kernel as one C function around the target's own instruction statements.

## Direction (user decision, 2026-09-14)

No per-target recipes or hand-written legality rules. The current Gemmini binding
(`experiments/inputs/target-inputs/gemmini/backend/gemmini_sched.py`: a LOOP_WS matmul recipe and rules read off
`LoopMatmul.scala`) is measured ONCE to validate IR -> C -> harness -> GSIM, then replaced by:

- recipes here, over abstract tile operations (load a tile into an operand store, multiply-accumulate
  a block, read out through the epilogue), sized from `targetgen.address_space` capacities;
- per-target instruction SEMANTICS in those terms, instantiated from derived facts and header encodings;
- FSM macro-instructions defined as loop nests of primitives and substituted by a generic `replace`;
- legality from interpreting the semantics (G1); only true hidden state declared per target.

## Invariants

- **Target-neutral.** No module here names a target or bakes in a target fact. Readout facts,
  capacities and instruction semantics arrive as data derived from the target's own sources, passed
  in by the caller.
- **A contract is compared by digest, never by name.** A program is graded against the golden of the
  contract whose `digest()` it declares. Provenance (which header the facts came from) is recorded
  but is not part of the digest.
- **Fail closed.** An unknown granularity, schema, dtype or rounding raises `ContractError`. It never
  falls back to a default. A declared-but-unfixed arithmetic (e.g. `rank1`) refuses rather than
  guesses.

## Added since (2026-09-18)

- `cost.py` — prices a schedule AND reports the kind of claim the number is: `exact`, `lower_bound`
  (some declared cost is a floor measured on an idle machine), or `undecidable` (some instruction has
  no derived cost on its unit). The last never returns a partial sum: the missing terms are unbounded,
  so a sum of the known ones is smaller than the truth by an unknown amount and would rank a schedule
  of unmeasured instructions as the cheapest available.
- `check/placement.py`, `check/sync.py` — the two machine-aware checks. What a wait ORDERS is declared
  per instruction, and only `completion` discharges a dependence on a result.
- `expressiveness.py` — measures whether an expert corpus and our own schedule AGREE on the capability
  axes `src/merlin/resources/contracts/schedule_ir_coverage.yaml` declares, and reports the answer as a TRIPLE
  (`expressed / denominator / unexercised`) rather than a ratio. Every field the comparator could not
  decide is charged to the denominator, so thin evidence reads as a wide interval; a corpus that does
  not resolve reports UNMEASURED naming the environment variable, path or audit it needs, never a zero
  denominator. Corpora are declared in `src/merlin/resources/contracts/corpora.yaml` (`sched_corpora`) and located
  through the hardware pin they name, so a result records which revision it came from. Run it with
  `build_tools/checks/check_schedule_ir_coverage.py --measure --summary`.

## Former ir/ (now `ir.py`)

## Purpose

The kernel IR `mk`: one fused group's schedule as counted loops over a target's schedule instructions.
A kernel names its DRAM tensor arguments and calls instructions whose operands are index expressions
over loop variables, byte pointers into those tensors, `NULL`, or floats.

## Files

- `expr.py` — index expressions: constants, loop variables, sums, products by a constant, and a
  `select` on one loop variable (last-tile remainders). Evaluate, render to IR text or C.
- `kernel.py` — `TensorArg`, `Ptr`, `NULL`, `Call`, `Loop`, `Kernel`; the canonical text form and its
  digest; target-free structural checks; enumeration of dynamic instances; per-instance concrete values.

## Invariants

- **The text form is the identity.** `Kernel.digest()` hashes `Kernel.text()`; floats print as exact hex.
  Two schedules are the same exactly when they print the same.
- **Target-neutral.** Nothing here knows an instruction. Instruction names are resolved against a
  `merlin.compiler.kernel_ir.ir.InstructionSet` by the checker and the emitter.
- **Operands stay small.** Anything a `select` on one variable cannot say is a loop transformation
  (split, peel), not a richer operand form.

## Former isa/ (now `ir.py`)

## Purpose

The shape of a target's schedule instruction set: `Operand`, `InstrDef`, `InstructionSet`, and
`instruction_set(target)`, which asks the target's backend hook `sched_instruction_set()` for it.

## Invariants

- `types.py` owns the pure declaration objects. The installed namespace
  reexports those exact objects and retains its backend lookup bridge. Shared
  source releases omit backend discovery rather than exporting runtime access.
- **Derived by the target, defined here.** This package holds no instruction. A target builds its
  instruction set from its own sources (header macros, RTL facts) and records their digests in
  `provenance`.
- **Checks see concrete values.** `check(values, state)` and `footprint(values)` receive one dynamic
  instance's evaluated operands; `state` is shared across one kernel's instances in program order, for
  rules that span instructions.
- **Identity.** `InstructionSet.digest()` covers operand lists, facts and provenance; a measurement
  keyed on a schedule also keys on this.

## Former primitives/ (now `primitives.py, loops.py`)

## Purpose

The scheduling language: the moves a schedule is written in. `merlin.compiler.kernel_ir.ir` says what a kernel does
and `merlin.compiler.kernel_ir.machine` says what the hardware is; this is the vocabulary that turns one kernel into
another without changing what it computes.

## Files

- `__init__.py` — `Cursor`, the two failure kinds, `proof_of`, and the primitives themselves
  (`divide_loop`, `unroll`, `reorder`, `fuse`), plus the shared obligation every rewrite discharges.

## Invariants

- **A failure has two kinds.** `NotApplicable` means this move does not apply here, so a search may try
  the next one; `PrimitiveError` means the schedule is wrong, so it must stop. It is a subclass, so a
  caller that does not know the difference still fails safe.
- **A cursor is a path, re-resolved at apply time.** A kernel's identity is its text, so a live handle
  into one means nothing after the first rewrite. A cursor that resolves to a different instruction than
  it asserted is a DEFINITE failure: the caller reasoned about another statement.
- **Every primitive declares its obligation, and the declaration is read, not authored.** The coverage
  register takes each axis' obligation from the primitive that carries it, so a row cannot claim a
  stronger one than the code discharges.
- **The shared obligation is the instance multiset, never the order.** A rewrite must preserve WHICH
  dynamic instances run with which operand values; changing when they run is the whole point. That is
  what catches the dropped, duplicated or mis-indexed instance a hand-written index rewrite produces.
- **Refuse rather than invent.** A split that would need a tail, an imperfect nest, two loops of
  differing extents: each refuses. Inventing the missing piece is how a schedule silently stops
  computing what it did.
- **Target-neutral.** Nothing here names a target. What a machine allows comes from `merlin.compiler.kernel_ir.machine`,
  which is derived from the target's own data.

## Former mach/ (now `machine.py`)

## Purpose

The machine a schedule is written against. `merlin.compiler.kernel_ir.ir` models one instruction stream, one agent
and no time, which is a faithful model of exactly one machine shape. This package adds what the other
shapes need and nothing more: units that issue and execute, the queue each consumes, latency per
*(instruction, unit)* pair, banked memories with ports and an arbiter, a thread hierarchy, and whether
hazards are resolved by hardware or by the compiler.

It is not a scheduling language. The primitives are `merlin.compiler.kernel_ir.primitives`; the transformation
scripting layer is the `transform` dialect. This is the machine those two are written against.

## Files

- `model.py` — `Machine` and its elements: `Unit`, `Latency`, `Memory`, `Level`/`Hierarchy`, `Unknown`;
  the closed vocabularies `UNIT_KINDS`, `HAZARD_RESOLUTIONS`, `COMPLETION_KINDS`; the overlap queries
  `can_overlap` / `shares_issue` / `resource_of`; and the canonical text form and its digest.

## Invariants

- **A machine is derived, never declared.** A target contributes data — its compute-unit contract, its
  RTL facts, its address space, its declared issue-scheduling rules. It never contributes a `Machine`.
  Nothing here names a target, and `build_tools/checks/check_no_target_name.py` covers this package.
- **The three-state rule is load-bearing.** An underived quantity is `None` **and** carries an
  `Unknown` saying why — never `0`, never a plausible default. `Machine.__post_init__` refuses a
  `hazard_resolution` of `None` with no matching `Unknown`, because assuming `interlocked` makes every
  illegal schedule look legal and assuming `explicit` makes a correctness gate that cannot fire look
  like one that can.
- **Latency belongs to the pair, not the instruction.** The same logical operation costs 95 cycles on
  one matrix unit and 34 on the other, so a single `latency` attribute cannot express it. `issue` and
  `result` are separate fields: a declared issue *distance* in instruction slots is not a completion
  latency, and conflating them prices a schedule wrongly.
- **The unit vocabulary is imported, not restated.** `UNIT_KINDS` is `compute_units.KINDS | MOVER_KINDS`,
  so the compute-unit taxonomy and this one cannot drift.
- **Provenance is inside the digest.** Two machines derived from different RTL are different machines
  even when every number matches, and a measurement keyed on one must not be reused for the other.

## Former check/ (now `checks.py`)

## Purpose

Correctness gates that need no simulator. They run on every schedule or contract change, before a
candidate spends RTL time.

## Modules

- `placement.py` — whether a kernel's on-chip placement is sound: a result written over an operand
  still being read, and a bank two contending engines share under a declared arbiter. Unlike `sync.py`
  this asks ONE question of every machine, because neither failure depends on the archetype.
- `sync.py` — whether a kernel's synchronisation is sound ON A GIVEN MACHINE. The same kernel is
  correct on one archetype and wrong on the other with no instruction changed, so this takes a
  `merlin.compiler.kernel_ir.machine.Machine` and asks the question that machine poses.
- `epilogue_enum.py` — exhaustive epilogue enumeration (gate G1e). It evaluates two elementwise
  readouts on every accumulator value in a range and reports each disagreement.
  `reachable_accumulator_range` bounds the range from the reduction length and the operand ranges.

## Invariants

- **Enumerate, do not sample.** The accumulator range of a real reduction is small enough to
  enumerate outright; a K=4608 int8 reduction reaches about 1.5e8 values. A sampled stimulus is how a
  dropped sign or a wrong tie rounding has passed before.
- **Neither side is trusted.** A `FlipReport` states where two readouts differ. Whether that is a
  defect or a declared bound is the caller's decision. `exact` is true only if every value in the
  range was evaluated and none differed.
- **The archetype decides the question, not just the answer.** Where hardware tracks dependencies an
  absent wait is not a defect and no correctness check can refute a reordering; where the compiler
  separates hazards the identical omission returns wrong data. A checker that asked one question would
  be wrong on half the corpus, and wrong in the dangerous direction on the half it passed.
- **Undecidable is a state, and its empty problem list is not a clean bill.** A machine that declares no
  hazard model gets neither answer: assuming hardware tracking accepts every illegal schedule, assuming
  compiler responsibility reports defects that may not exist. `SyncReport.decidable` is the field that
  says whether the question was asked at all.

## Former contract/ (now `checks.py`)

## Purpose

Numerics contracts: how an integer accumulator becomes the stored output. A contract fixes the scale
granularity (`tensor` / `row` / `column` / `rank1`), the rounding, the clamp bounds and the admitted
activations. Goldens are generated under a contract, and grading compares the contract digest.

## Modules

- `registry.py` — `NumericsContract` (with `readout()` and `digest()`), `from_readout_facts()` and the
  readout-level registry (`per_tensor_readout_v1`, `per_row_readout_v1`, `per_column_readout_v1`).

## Invariants

- **Facts in, never literals.** `from_readout_facts` consumes the dict a target's header-verified
  readout contract returns (schema `scalar_narrow_readout_contract_v1`). The rounding is declared
  `half_even` only because that schema is produced after the target's scale and rounding macros were
  matched against the round-half-to-even construction.
- **The arithmetic is the C macro's, bit for bit**: one f32 multiply, `rint`, clamp, optional relu.
  `tests/ir/test_sched_contract.py` holds it to `runtime.tensor.Tensor.requant_acc_scale`.
- **Readout-level only.** Bias seeding, residual add and pooling are further stages composed by
  model-level contracts. They are not smuggled in here.

## Former codegen/ (now `codegen.py`)

## Purpose

Emit a `mk` kernel as one C function: counted loops around each call's statement, which the target's
`InstrDef.render_c` produces. Pointers render as byte offsets from their tensor argument, floats as exact
float32 hex literals, and the body ends with the instruction set's `drain_c`.

## Invariants

- **No instruction knowledge.** Every statement text comes from the target's instruction set.
- **Refuse, never guess.** An operand list that differs from the instruction's, or a dtype with no C
  type, raises `IsaError`.
- **Version.** `EMITTER_VERSION` changes whenever the text emitted for an unchanged kernel changes;
  measurements key on it.
