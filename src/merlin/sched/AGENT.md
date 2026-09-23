# AGENT.md — merlin/python/merlin/sched

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
(`merlin/targets/gemmini/backend/gemmini_sched.py`: a LOOP_WS matmul recipe and rules read off
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
  axes `merlin/contract/schedule_ir_coverage.yaml` declares, and reports the answer as a TRIPLE
  (`expressed / denominator / unexercised`) rather than a ratio. Every field the comparator could not
  decide is charged to the denominator, so thin evidence reads as a wide interval; a corpus that does
  not resolve reports UNMEASURED naming the environment variable, path or audit it needs, never a zero
  denominator. Corpora are declared in `merlin/contract/corpora.yaml` (`sched_corpora`) and located
  through the hardware pin they name, so a result records which revision it came from. Run it with
  `build_tools/scripts/check_schedule_ir_coverage.py --measure --summary`.
