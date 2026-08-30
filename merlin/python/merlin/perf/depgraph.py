"""Dependence over an emitted schedule: what must be separated, by how much, and what that bounds.

WHAT THIS IS FOR. Everything else in this package accounts for BUSY time -- how much work a schedule
asks of each engine, and how long each engine needs for it. Measured on a real kernel, three quarters
of the cycles had no engine busy at all, which means the accounting is structurally blind to most of
the runtime. Those cycles are separations: the machine is waiting, because it has no interlock and
the compiler inserted a wait. This module is about the waits.

WHY AN EDGE HERE IS A DIFFERENCE, NOT A COST. An edge weight is a REQUIRED SEPARATION: the number of
cycles that must elapse between one instruction and another. On a machine with no interlock that is
architecturally real -- reading a result early returns stale bytes and a vector op issued too soon
does not write its destination at all -- and it is genuinely pairwise. It is also the right shape for
evidence this thin: a separation SURVIVES AN UNKNOWN ABSOLUTE RATE. If a unit's initiation interval
is UNKNOWN but the same edge appears in both candidates, the unknown cancels out of the difference.
That is :mod:`merlin.perf.differential`'s theorem lifted from resources to edges, and it is why
schedules can be ranked on a target where almost nothing can be priced end to end.

WHAT THIS IS NOT. It is not a scheduler: it scores schedules the emitter already produces and never
emits an instruction order. It is also not the all-to-all formulation it looks like -- reading the
matrix as "accumulate a weight while visiting every node" is the Sequential Ordering Problem, which
is NP-hard and stays so under pruning. The tractable reading of the SAME structure is a DAG of real
dependences and a LONGEST PATH over it, which is ``O(V+E)`` and needs no search.

THE THREE NUMBERS, and they answer different questions:

* ``as_emitted`` -- what this ordering costs: the in-order chain plus every separation it did not
  cover with work. This is what the machine does today.
* ``critical_path`` -- the longest chain of real dependences with the program order REMOVED. No legal
  reordering of these instructions can finish sooner, so it is an exact LOWER BOUND on the makespan
  and the first number here that speaks to the idle cycles.
* the gap between them -- the separation cycles that are covered by nothing. That is the quantity a
  reordering pass would spend its effort on, and it is measured rather than asserted.

WHAT STAYS UNKNOWN, deliberately. A separation whose length is not derivable gets weight ZERO in the
resolved path and is recorded as an exposed unknown EDGE CLASS with a count. Weight zero is the only
choice that keeps the resolved path a valid lower bound (a separation can only add), and the count is
exactly what :func:`merlin.perf.differential.comparable` checks before it will difference two
candidates. An initiation interval is not invented here, and a flat per-operation latency table is
not consulted: a table that conflates an initiation interval with a completion latency prices a
loop-carried edge wrong in the direction that reads as "the corpus under-delays".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from merlin.perf.decompose import UNKNOWN
from merlin.perf.deps.liveness import (Access, Effects, Instruction, effects_of, liveness,
                                       pressure, value_ranges)
from merlin.perf.envelope import Composed
from merlin.perf.headroom import Composition

__all__ = [
    "Edge", "Dag", "CriticalPath", "IssueModel", "Program", "Region",
    "decode_program",
    "RAW", "WAR", "WAW", "SEPARATION",
    "build_dag", "critical_path", "makespan", "to_composed", "demands_of",
    "probe_issue_model", "program_from_plan", "candidates_for", "analyse_program",
]

RAW = "raw"          # a consumer reads what a producer wrote
WAR = "war"          # a writer must not clobber what a reader has not read yet
WAW = "waw"          # two writers of the same value must keep their order
SEPARATION = "separation"   # a wait the schedule itself declares, in cycles


@dataclass(frozen=True)
class Edge:
    """One required separation between two instructions."""

    src: int
    dst: int
    kind: str
    #: Cycles that must elapse, or UNKNOWN when the machine does not publish the latency. UNKNOWN is
    #: a distinct inhabited state; it is read as zero ONLY when computing a lower bound, and the fact
    #: that it was is carried in :attr:`edge_class`.
    cycles: "float | Any"
    #: The class an UNKNOWN edge belongs to, so identical unknowns across two candidates can cancel.
    edge_class: str = ""
    value: Access | None = None

    @property
    def known(self) -> bool:
        return self.cycles is not UNKNOWN

    @property
    def weight(self) -> float:
        """The weight a LOWER BOUND may use: the separation where it is known, zero where it is not.

        Zero rather than a guess, and only ever downward: a separation can only push two instructions
        further apart, so ignoring one keeps the path a bound instead of turning it into a claim."""
        return float(self.cycles) if self.known else 0.0


@dataclass(frozen=True)
class IssueModel:
    """How long the sequencer itself takes, MEASURED rather than assumed.

    ``issue_cycles`` is what one instruction costs the in-order front end, and ``stall_unit`` is what
    one unit of a stall instruction's immediate costs. Both are differences -- the slope of cycles
    against a count -- so neither depends on knowing what the rest of the program costs. The tier they
    were measured on travels with them, because two runners for the same machine have been measured
    to disagree about its control flow, and there is no reason to expect them to agree about this.
    """

    issue_cycles: float
    stall_unit: float
    tier: str
    provenance: str

    def cost_of(self, instruction: Instruction, stall_mnemonic: str, stall_operand: str = "imm") -> float:
        if instruction.mnemonic == stall_mnemonic:
            return self.stall_unit * float(instruction.operands.get(stall_operand, 0))
        return float(self.issue_cycles)


@dataclass(frozen=True)
class Dag:
    """A region's instructions and the separations between them."""

    instructions: tuple[Instruction, ...]
    effects: tuple[Effects, ...]
    edges: tuple[Edge, ...]
    node_cycles: tuple[float, ...]
    #: Instructions whose dependence footprint was incomplete. Every path below is conditional on
    #: these: an unresolved operand is a MISSING edge, and a missing edge is a separation a reordering
    #: would delete.
    incomplete: tuple[str, ...] = ()

    def exposed_classes(self) -> dict[str, int]:
        """``{edge class: how many edges of it are unpriced}`` -- the demand side of the comparison."""
        out: dict[str, int] = {}
        for e in self.edges:
            if not e.known:
                out[e.edge_class] = out.get(e.edge_class, 0) + 1
        return out


@dataclass(frozen=True)
class CriticalPath:
    """The longest chain of separations through a region, and what it left out."""

    cycles: float
    path: tuple[int, ...]
    exposed: Mapping[str, int]
    #: True when every edge on the path had a known weight, so the number is the chain's real length
    #: rather than a floor.
    complete: bool
    incomplete: tuple[str, ...] = ()

    def claim(self) -> str:
        if self.complete and not self.incomplete:
            return f"{self.cycles:.0f} cycles along {len(self.path)} instruction(s)"
        exposed = ", ".join(f"{k}x{v}" for k, v in sorted(self.exposed.items())) or "none"
        return (f"AT LEAST {self.cycles:.0f} cycles along {len(self.path)} instruction(s); "
                f"unpriced separations on the graph: {exposed}")


# ---------------------------------------------------------------------------------------------------
# building the graph
# ---------------------------------------------------------------------------------------------------
def _separation_class(instruction: Instruction, roles: Mapping[str, str] | None) -> str:
    """Which unknown a producer's completion latency belongs to.

    Grouped by the producer's DERIVED structural role rather than by its mnemonic, because that is the
    granularity at which the unknown is the same quantity: two multiplies into the same array wait on
    the same pipeline. Grouping too finely would leave two candidates with different-looking unknown
    sets and refuse a comparison that is actually sound; grouping too coarsely would let two genuinely
    different latencies cancel when they should not."""
    role = (roles or {}).get(instruction.mnemonic)
    return f"{SEPARATION}.{role or instruction.mnemonic}"


def build_dag(instructions: "Sequence[Instruction]", effects: "Sequence[Effects]", *,
              issue: IssueModel, stall_mnemonic: str, stall_operand: str = "imm",
              roles: Mapping[str, str] | None = None,
              resolved_separations: Mapping[str, float] | None = None,
              memory_conflicts: "Sequence[tuple[int, int, str]]" = ()) -> Dag:
    """Every real dependence between these instructions, with the separation each one requires.

    Edges come from three places and nothing else:

    * register dependences -- read-after-write, write-after-read and write-after-write over the values
      the MEASURED operand-direction model says each instruction defines and uses. No operand name is
      interpreted; a value is a state file and a slot, both of which the probe observed;
    * the schedule's own declared waits -- a stall instruction placed after a producer is that
      schedule's statement of how long its consumers must be held off, and it is a KNOWN weight
      because the machine publishes the immediate and the sequencer honours it. This is what makes
      the resolved part of the path non-trivial on a target whose unit latencies are all unknown;
    * memory-region conflicts the caller computed from the program's own placements.

    A read-after-write whose separation the schedule did NOT declare gets UNKNOWN, classed by the
    producer's role. It is never given a latency from a table: the shipped flat per-operation latency
    for this class of unit conflates an initiation interval with a completion latency, and an edge
    built on that is wrong in the direction that reads as "the schedule under-delays".
    """
    resolved = dict(resolved_separations or {})
    node_cycles = tuple(issue.cost_of(i, stall_mnemonic, stall_operand) for i in instructions)

    # A stall following instruction P is P's declared separation from its consumers.
    declared: dict[int, float] = {}
    for index, instruction in enumerate(instructions):
        if instruction.mnemonic != stall_mnemonic or index == 0:
            continue
        producer = index - 1
        declared[producer] = declared.get(producer, 0.0) + node_cycles[index]

    last_def: dict[Access, int] = {}
    readers: dict[Access, list[int]] = {}
    edges: list[Edge] = []
    carried: set[int] = set()          # stalls whose cycles an edge now carries

    def _sep(src: int, value: Access | None, kind: str) -> Edge:
        cls = _separation_class(instructions[src], roles)
        if src in declared:
            return Edge(src, 0, kind, declared[src], f"{cls}.declared", value)
        if cls in resolved:
            return Edge(src, 0, kind, float(resolved[cls]), cls, value)
        return Edge(src, 0, kind, UNKNOWN, cls, value)

    for index, effect in enumerate(effects):
        for value in effect.uses:
            producer = last_def.get(value)
            if producer is not None:
                proto = _sep(producer, value, RAW)
                edge = Edge(producer, index, RAW, proto.cycles, proto.edge_class, value)
                if edge not in edges:
                    # Two operands of one instruction reading the SAME value is one dependence, not
                    # two. Counting it twice inflates the demand on an unpriced class, and the demand
                    # is what decides whether two candidates may be differenced at all.
                    edges.append(edge)
                if producer in declared:
                    carried.add(producer)
            readers.setdefault(value, []).append(index)
        for value in effect.defs:
            producer = last_def.get(value)
            if producer is not None:
                edges.append(Edge(producer, index, WAW, 0.0, f"{WAW}.ordering", value))
            for reader in readers.get(value, ()):
                if reader < index:
                    edges.append(Edge(reader, index, WAR, 0.0, f"{WAR}.ordering", value))
            readers[value] = []
            last_def[value] = index
    for src, dst, why in memory_conflicts:
        edges.append(Edge(src, dst, WAW, UNKNOWN, f"memory.{why}", None))

    # A stall whose cycles an edge now carries must NOT also be charged as a node: its time is the
    # separation between the producer and its consumer, and paying for it in both places prices every
    # wait twice and makes moving work into a wait shadow save nothing. A stall guarding a producer
    # with no observable definition keeps its cycles as a node, because nothing else is accounting
    # for them -- which is the honest treatment of a wait whose purpose the probe could not see.
    node_cycles = tuple(0.0 if (instructions[i].mnemonic == stall_mnemonic and i - 1 in carried)
                        else c for i, c in enumerate(node_cycles))
    incomplete = tuple(f"[{i}] {ins.mnemonic}: {'; '.join(eff.unresolved)}"
                       for i, (ins, eff) in enumerate(zip(instructions, effects)) if eff.unresolved)
    return Dag(instructions=tuple(instructions), effects=tuple(effects), edges=tuple(edges),
               node_cycles=node_cycles, incomplete=incomplete)


def critical_path(dag: Dag) -> CriticalPath:
    """The longest chain of dependences, with the PROGRAM ORDER removed -- an exact lower bound.

    Removing the program order is the point. With it, the longest path is just the schedule the
    machine already runs and says nothing new. Without it, the path is the longest chain no
    reordering can break, so nothing that keeps these instructions and these dependences can finish
    sooner. That is a bound rather than a prediction, and it is the number the idle cycles have to be
    argued against.

    ``O(V+E)``: one forward sweep in index order, which is a topological order because every edge
    joins an earlier instruction to a later one.
    """
    n = len(dag.instructions)
    preds: list[list[Edge]] = [[] for _ in range(n)]
    for e in dag.edges:
        if 0 <= e.src < n and 0 <= e.dst < n and e.src < e.dst:
            preds[e.dst].append(e)
    best = [0.0] * n
    from_who: list[int | None] = [None] * n
    all_known = True
    for j in range(n):
        top = 0.0
        who: int | None = None
        for e in preds[j]:
            if not e.known:
                all_known = False
            cand = best[e.src] + e.weight
            if cand > top:
                top, who = cand, e.src
        best[j] = top + dag.node_cycles[j]
        from_who[j] = who
    if not n:
        return CriticalPath(0.0, (), {}, True, dag.incomplete)
    end = max(range(n), key=lambda i: best[i])
    path: list[int] = []
    cur: int | None = end
    while cur is not None:
        path.append(cur)
        cur = from_who[cur]
    return CriticalPath(cycles=best[end], path=tuple(reversed(path)),
                        exposed=dag.exposed_classes(), complete=all_known,
                        incomplete=dag.incomplete)


def makespan(dag: Dag, order: "Sequence[int]") -> float:
    """What one ORDERING of these instructions costs on an in-order, non-interlocked sequencer.

    The sequencer issues in order, so the ordering contributes a chain whose edges are each
    instruction's own issue cost. Every declared separation still applies between the SAME two
    instructions wherever they now sit, so work moved between a producer and its consumer counts
    against the wait instead of being paid on top of it. That is the entire mechanism by which
    hoisting a transfer out of a wait shadow saves anything, and modelling it any other way makes
    every reordering look free and every schedule look identical.
    """
    position = {index: slot for slot, index in enumerate(order)}
    n = len(order)
    preds: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for e in dag.edges:
        a, b = position.get(e.src), position.get(e.dst)
        if a is None or b is None:
            continue
        if a < b:
            preds[b].append((a, e.weight))
        elif a > b:
            # The ordering violates this dependence. That is not a schedule; refusing is the only
            # honest answer, because pricing an illegal order produces a number that beats every
            # legal one.
            raise ValueError(f"the proposed order violates a {e.kind} dependence between "
                             f"instruction {e.src} and {e.dst}")
    start = [0.0] * n
    for slot in range(n):
        top = start[slot - 1] + dag.node_cycles[order[slot - 1]] if slot else 0.0
        for a, w in preds[slot]:
            top = max(top, start[a] + dag.node_cycles[order[a]] + w)
        start[slot] = top
    return start[-1] + dag.node_cycles[order[-1]] if n else 0.0


# ---------------------------------------------------------------------------------------------------
# the composed bound and the ranking
# ---------------------------------------------------------------------------------------------------
def demands_of(dag: Dag) -> dict[str, float]:
    """How much work each UNRESOLVED separation class is asked for -- the count of exposed edges.

    Counted over the WHOLE graph rather than over the critical path. Two orderings of the same
    instructions expose exactly the same unpriced edges, so a whole-graph count is equal between them
    and their unknowns cancel; a path-restricted count would differ between candidates and refuse a
    comparison that is sound. It is also the conservative direction: an unpriced edge that turns out
    not to be critical still had to be counted, never wished away."""
    return {name: float(count) for name, count in dag.exposed_classes().items()}


def to_composed(cycles: float, dag: Dag, *, floor: float | None = None) -> Composed:
    """Package a resolved cycle count as a :class:`~merlin.perf.envelope.Composed`.

    ``partial_cycles`` is the resolved part, ``unresolved`` names the separation classes that were not
    priced, and ``cycles`` stays UNKNOWN whenever any of them is exposed -- a total that quietly
    becomes a number is the failure this whole layer exists to prevent. The operator is SUM with no
    overlap credit, which is not a default but a statement about this machine: the sequencer is
    in-order and a separation is time during which it issues nothing, so separations add.
    """
    exposed = tuple(sorted(dag.exposed_classes()))
    return Composed(cycles=UNKNOWN if exposed else float(cycles), partial_cycles=float(cycles),
                    floor_cycles=float(cycles if floor is None else floor),
                    operator=Composition.SUM, eta=0.0, overlap_saving=0.0,
                    unresolved=exposed, workload_fixed_cycles=0)


# ---------------------------------------------------------------------------------------------------
# measuring the sequencer
# ---------------------------------------------------------------------------------------------------
def probe_issue_model(run_kernel, build_padding, *, tier: str, counts=(4, 64),
                      stalls=(0, 256)) -> IssueModel:
    """MEASURE what one instruction and one stall unit cost, by differencing two programs.

    ``build_padding(n_instructions, stall_cycles) -> kernel source`` emits a program with the given
    amount of padding and nothing else; ``run_kernel(source) -> cycles`` runs it. Two points per
    parameter, because one point cannot separate a rate from an intercept -- the same rule that
    governs every fitted term in this package.

    ONE REQUIREMENT ON ``build_padding``, and it is not cosmetic: the stall must have at least one
    real instruction AFTER it, before the terminator. Measured on the RTL tier here, a stall placed
    immediately before the terminator costs ONE cycle whatever its immediate says -- the machine
    finishes rather than waiting -- while the same stall with any instruction behind it costs its
    immediate exactly. A probe that puts the stall last therefore measures a stall unit near zero,
    concludes that every wait in the program is free, and reports a schedule with nothing to gain.

    Raises rather than defaulting. A sequencer given an assumed issue rate turns every schedule
    comparison into a comparison of that assumption.
    """
    lo_n, hi_n = int(counts[0]), int(counts[-1])
    lo_s, hi_s = int(stalls[0]), int(stalls[-1])
    c_lo_n = run_kernel(build_padding(lo_n, lo_s))
    c_hi_n = run_kernel(build_padding(hi_n, lo_s))
    c_hi_s = run_kernel(build_padding(lo_n, hi_s))
    for name, value in (("instruction sweep low", c_lo_n), ("instruction sweep high", c_hi_n),
                        ("stall sweep high", c_hi_s)):
        if value is None:
            raise ValueError(f"the {name} probe did not halt, so the sequencer's cost is UNKNOWN")
    issue = (float(c_hi_n) - float(c_lo_n)) / max(1, hi_n - lo_n)
    unit = (float(c_hi_s) - float(c_lo_n)) / max(1, hi_s - lo_s)
    return IssueModel(issue_cycles=issue, stall_unit=unit, tier=tier, provenance=(
        f"measured on tier {tier}: {hi_n - lo_n} extra instructions cost {c_hi_n - c_lo_n} cycles "
        f"({issue:.3f}/instruction); {hi_s - lo_s} extra stall units cost {c_hi_s - c_lo_n} cycles "
        f"({unit:.3f}/unit)"))


# ---------------------------------------------------------------------------------------------------
# a generated plan, decoded back into a dependence graph
# ---------------------------------------------------------------------------------------------------
@dataclass
class Region:
    """A straight-line stretch of an emitted program, and how many times it runs."""

    name: str
    start: int
    end: int                       # exclusive
    trips: int = 1

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass
class Program:
    """A decoded emitted program: instructions, their measured effects, and its loop structure."""

    instructions: tuple[Instruction, ...]
    effects: tuple[Effects, ...]
    regions: tuple[Region, ...]
    roles: Mapping[str, str] = field(default_factory=dict)
    capacities: Mapping[str, int] = field(default_factory=dict)


def decode_program(isa: Any, words: "Sequence[int]") -> list[tuple[str | None, dict[str, int]]]:
    """Decode a word stream back to ``(mnemonic key, operands)`` using the target's own signatures.

    The disassembler proper reports an instruction's SEMANTIC CLASS, which is what a coverage report
    wants; a dependence graph wants the key the direction model is keyed by, and the two are not the
    same map. A whole family of transfer instructions shares one class while each channel is its own
    key, so classifying by class would look up the direction of an instruction nobody probed and
    quietly return nothing. So this decodes to the key, by matching each op's derived
    (fixed_mask, fixed_value) signature -- the same legality oracle, read for identity.

    A word that matches no signature decodes to ``None``: an invented or garbled encoding, which is a
    real answer and becomes an instruction whose dependence footprint is unresolved."""
    entries = list((isa.by_mnemonic or {}).items())
    out: list[tuple[str | None, dict[str, int]]] = []
    for raw in words:
        word = int(raw) & 0xFFFFFFFF
        found: tuple[str, dict] | None = None
        for key, ent in entries:
            mask, value = ent.get("fixed_mask"), ent.get("fixed_value")
            if not (isinstance(mask, int) and isinstance(value, int)) or (word & mask) != value:
                continue
            operands: dict[str, int] = {}
            for attr, bits in (ent.get("fields") or {}).items():
                acc = 0
                for i, word_bit in enumerate(bits):
                    if isinstance(word_bit, int) and word_bit >= 0 and (word >> word_bit) & 1:
                        acc |= 1 << i
                operands[attr] = acc
            found = (key, operands)
            break
        out.append(found if found is not None else (None, {}))
    return out


def program_from_plan(plan: Any, directions: Any) -> Program:
    """Decode a generated plan's OWN emitted words back into instructions, and attach their effects.

    Decoding the words rather than remembering what was emitted is deliberate: the graph is then built
    from the bytes the machine will actually execute, so an encoding mistake shows up as an
    instruction that decodes to nothing rather than as a graph that silently describes a different
    program. The branch contract used to resolve a branch's target is the plan's own MEASURED one --
    a branch immediate is not portable knowledge, and reading one as a byte offset produces a program
    whose loops never close while every instruction still decodes.
    """
    isa = plan.facts.isa
    branch_mnemonics = {plan.ops.branch_ne}
    scale = max(1, int(plan.control_flow.branch_imm_scale))
    instructions: list[Instruction] = []
    for index, (mnemonic, operands) in enumerate(decode_program(isa, list(plan.words))):
        target = None
        if mnemonic in branch_mnemonics and "imm" in operands:
            width = len(isa.fields_of(mnemonic).get("imm") or ())
            raw = operands["imm"]
            signed = raw - (1 << width) if width and raw >= (1 << (width - 1)) else raw
            target = index + signed // scale
        instructions.append(Instruction(index=index, mnemonic=mnemonic or "", operands=operands,
                                        branch_target=target))
    effects = tuple(effects_of(i, directions) for i in instructions)

    # Regions: the straight-line stretches between loop boundaries. A backward branch at ``b`` to
    # ``t`` closes a loop whose body is [t, b]; everything outside every loop runs once.
    loops = [(int(i.branch_target), i.index) for i in instructions if i.branches_backward]
    cut = {0, len(instructions)}
    for target, branch in loops:
        cut.add(max(0, target))
        cut.add(min(len(instructions), branch + 1))
    bounds = sorted(c for c in cut if 0 <= c <= len(instructions))
    regions = tuple(Region(name=f"[{a},{b})", start=a, end=b, trips=1)
                    for a, b in zip(bounds, bounds[1:]) if b > a)
    roles = {mn: str((ent or {}).get("role") or "") for mn, ent in (isa.by_mnemonic or {}).items()}
    return Program(instructions=tuple(instructions), effects=effects, regions=regions, roles=roles)


# ---------------------------------------------------------------------------------------------------
# the three candidate schedules
# ---------------------------------------------------------------------------------------------------
def _legal(dag: Dag, order: "Sequence[int]") -> bool:
    position = {index: slot for slot, index in enumerate(order)}
    return all(position[e.src] < position[e.dst] for e in dag.edges
               if e.src in position and e.dst in position)


def _list_schedule(dag: Dag, indices: "Sequence[int]", *, priority) -> list[int]:
    """List scheduling over the dependence DAG, highest priority ready instruction first.

    This is "pick the best next node" made tractable: ``O(E + V log V)`` rather than a search over
    orderings. It is a HEURISTIC and is reported as one -- its distance from the critical path is
    measurable and is reported beside it, and no claim of optimality is made anywhere."""
    inside = set(indices)
    preds: dict[int, set[int]] = {i: set() for i in indices}
    succs: dict[int, set[int]] = {i: set() for i in indices}
    for e in dag.edges:
        if e.src in inside and e.dst in inside and e.src != e.dst:
            preds[e.dst].add(e.src)
            succs[e.src].add(e.dst)
    remaining = dict((i, set(p)) for i, p in preds.items())
    ready = [i for i in indices if not remaining[i]]
    out: list[int] = []
    while ready:
        ready.sort(key=lambda i: (-priority(i), i))
        pick = ready.pop(0)
        out.append(pick)
        for s in succs[pick]:
            remaining[s].discard(pick)
            if not remaining[s] and s not in out and s not in ready:
                ready.append(s)
    return out if len(out) == len(list(indices)) else list(indices)


def _heights(dag: Dag, indices: "Sequence[int]") -> dict[int, float]:
    """Longest remaining path from each instruction -- the priority a critical-path list schedule uses."""
    inside = set(indices)
    succs: dict[int, list[Edge]] = {i: [] for i in indices}
    for e in dag.edges:
        if e.src in inside and e.dst in inside:
            succs[e.src].append(e)
    height: dict[int, float] = {}
    for i in sorted(indices, reverse=True):
        height[i] = dag.node_cycles[i] + max((e.weight + height.get(e.dst, 0.0)
                                              for e in succs[i]), default=0.0)
    return height


def candidates_for(dag: Dag, indices: "Sequence[int]", *, stall_mnemonic: str,
                   hoist_role: str | None, roles: Mapping[str, str]) -> dict[str, list[int]]:
    """The three schedules this ranking is about, each a REORDERING of the same instructions.

    They are deliberately the same multiset: identical work asked of every unpriced resource is
    exactly the condition :func:`merlin.perf.differential.comparable` checks before it will difference
    two bounds, so keeping the multiset fixed is what makes the comparison exact rather than refused.

    * ``as_emitted`` -- the order the generator produced.
    * ``movement_hoisted`` -- every instruction of the movement role pulled as early as its
      dependences allow, which pulls a transfer out of the shadow of the wait before it.
    * ``stalls_tightened`` -- a critical-path list schedule, which fills each declared wait with the
      independent work the graph says may sit inside it. The wait is not shortened below what it
      declares; the work simply happens during it, which is the same saving and needs no claim about
      a latency nobody has measured.
    """
    order = list(indices)
    out = {"as_emitted": order}
    if hoist_role:
        movement = {i for i in indices if roles.get(dag.instructions[i].mnemonic) == hoist_role}
        out["movement_hoisted"] = _list_schedule(
            dag, indices, priority=lambda i: (2.0 if i in movement else 0.0))
    heights = _heights(dag, indices)
    out["stalls_tightened"] = _list_schedule(dag, indices, priority=lambda i: heights.get(i, 0.0))
    return {k: v for k, v in out.items() if _legal(dag, v)}


def _region_report(program: Program, region: Region, *, issue: IssueModel, stall_mnemonic: str,
                   hoist_role: str | None,
                   resolved_separations: Mapping[str, float] | None) -> dict:
    """The same analysis, restricted to one straight-line region and reported per iteration.

    A looped kernel's static instruction list is not its schedule: the body between a backward branch
    and its target runs once per trip, and that body is where a reordering pays off, multiplied. So
    the region is analysed on its own and the report says what ONE iteration costs; multiplying by the
    trip count is the caller's, and is only done where the trip count is actually known."""
    ins = program.instructions[region.start:region.end]
    eff = program.effects[region.start:region.end]
    if not ins:
        return {"name": region.name, "instructions": 0}
    renumbered = tuple(Instruction(index=i, mnemonic=x.mnemonic, operands=x.operands,
                                   branch_target=None, section=region.name)
                       for i, x in enumerate(ins))
    dag = build_dag(renumbered, eff, issue=issue, stall_mnemonic=stall_mnemonic,
                    roles=program.roles, resolved_separations=resolved_separations)
    cp = critical_path(dag)
    indices = list(range(len(renumbered)))
    schedules = candidates_for(dag, indices, stall_mnemonic=stall_mnemonic, hoist_role=hoist_role,
                               roles=program.roles)
    costs = {name: makespan(dag, order) for name, order in schedules.items()}
    emitted = costs.get("as_emitted")
    best = min(costs.values()) if costs else None
    return {
        "name": region.name, "start": region.start, "end": region.end, "trips": region.trips,
        "instructions": len(renumbered),
        "as_emitted_cycles": emitted,
        "critical_path_cycles": cp.cycles,
        "critical_path_complete": cp.complete,
        "reorder_slack_cycles": (None if emitted is None else emitted - cp.cycles),
        "best_candidate_cycles": best,
        "candidate_saving_cycles": (None if emitted is None or best is None else emitted - best),
        "schedules": {k: costs[k] for k in sorted(costs)},
        "unpriced_by_class": dag.exposed_classes(),
    }


def analyse_program(program: Program, directions: Any, *, issue: IssueModel, stall_mnemonic: str,
                    hoist_role: str | None = None,
                    resolved_separations: Mapping[str, float] | None = None,
                    capacities: Mapping[str, int] | None = None,
                    measured_cycles: float | None = None) -> dict:
    """Liveness, pressure, the critical path, and the ranking -- for one decoded program.

    The report is deliberately verbose about what it could not establish. An unresolved operand, an
    unpriced separation class and a register file of unknown capacity each change what the numbers
    below may be used for, and a reader who cannot see them will read a floor as a prediction.
    """
    from merlin.perf import differential

    dag = build_dag(program.instructions, program.effects, issue=issue,
                    stall_mnemonic=stall_mnemonic, roles=program.roles,
                    resolved_separations=resolved_separations)
    live_in, live_out = liveness(program.instructions, program.effects)
    caps = dict(capacities or program.capacities or {})
    ranges = value_ranges(program.instructions, program.effects, live_out)
    press = pressure(live_in, caps)

    cp = critical_path(dag)
    indices = list(range(len(program.instructions)))
    schedules = candidates_for(dag, indices, stall_mnemonic=stall_mnemonic,
                               hoist_role=hoist_role, roles=program.roles)
    composed: dict[str, Composed] = {}
    costs: dict[str, float] = {}
    for name, order in schedules.items():
        costs[name] = makespan(dag, order)
        composed[name] = to_composed(costs[name], dag, floor=cp.cycles)
    demands = {name: demands_of(dag) for name in composed}
    ranking, refusals = differential.rank_schedules(composed, demands=demands)
    pairwise = []
    names = sorted(composed)
    for a_i, a in enumerate(names):
        for b in names[a_i + 1:]:
            c = differential.compare(composed[a], composed[b], demands_a=demands[a],
                                     demands_b=demands[b], label_a=a, label_b=b)
            pairwise.append({"a": a, "b": b, "basis": c.basis, "faster": c.faster,
                             "delta_cycles": c.delta_cycles, "claim": c.claim()})

    as_emitted = costs.get("as_emitted")
    report: dict = {
        "instructions": len(program.instructions),
        "issue_model": {"issue_cycles": issue.issue_cycles, "stall_unit": issue.stall_unit,
                        "tier": issue.tier, "provenance": issue.provenance},
        "edges": {"total": len(dag.edges),
                  "by_kind": {k: sum(1 for e in dag.edges if e.kind == k)
                              for k in (RAW, WAR, WAW)},
                  "unpriced_by_class": dag.exposed_classes()},
        "critical_path": {"cycles": cp.cycles, "complete": cp.complete,
                          "length_instructions": len(cp.path), "claim": cp.claim(),
                          "exposed": dict(cp.exposed)},
        "as_emitted_cycles": as_emitted,
        "reorder_slack_cycles": (None if as_emitted is None else as_emitted - cp.cycles),
        "schedules": {name: {"cycles": costs[name],
                             "delta_vs_as_emitted": (None if as_emitted is None
                                                     else costs[name] - as_emitted)}
                      for name in sorted(costs)},
        "ranking": ranking,
        "pairwise": pairwise,
        "refusals": [c.reason for c in refusals],
        "pressure": [{"file": p.file, "peak": p.peak, "at": p.at_index, "capacity": p.capacity,
                      "fits": p.fits, "claim": p.claim()} for p in press],
        "value_ranges": {"defined": len(ranges),
                         "escaping": sum(1 for r in ranges if r.escapes),
                         "never_read": sum(1 for r in ranges if r.last_use is None and not r.escapes),
                         "max_length": max((r.length for r in ranges), default=0)},
        "incomplete_instructions": list(dag.incomplete),
    }
    regions = [_region_report(program, r, issue=issue, stall_mnemonic=stall_mnemonic,
                              hoist_role=hoist_role, resolved_separations=resolved_separations)
               for r in program.regions]
    report["regions"] = regions
    known_trips = all(r.get("trips") is not None for r in regions)
    report["dynamic_cycles"] = (
        sum(float(r["trips"]) * float(r["as_emitted_cycles"]) for r in regions
            if r.get("as_emitted_cycles") is not None) if known_trips else None)
    report["dynamic_best_cycles"] = (
        sum(float(r["trips"]) * float(r["best_candidate_cycles"]) for r in regions
            if r.get("best_candidate_cycles") is not None) if known_trips else None)
    report["dynamic_critical_path_cycles"] = (
        sum(float(r["trips"]) * float(r["critical_path_cycles"]) for r in regions
            if r.get("critical_path_cycles") is not None) if known_trips else None)
    backward = sum(1 for i in program.instructions if i.branches_backward)
    report["loop_carried"] = {
        "backward_branches": backward,
        "modelled": False,
        "note": ("a dependence that crosses a backward branch -- the accumulate chain above all -- is "
                 "NOT an edge in this graph. Its separation is UNKNOWN and measured to exceed the "
                 "naive sum of the published per-operation latencies, so giving it a weight would be "
                 "fitting rather than deriving. Every per-region number below is therefore a bound on "
                 "ONE iteration, and the loop-carried separation is an additive term nobody here has "
                 "measured."),
    }
    if measured_cycles is not None and as_emitted is not None:
        report["measured_cycles"] = measured_cycles
        report["bound_vs_measured"] = {
            "critical_path_over_measured": cp.cycles / measured_cycles if measured_cycles else None,
            "as_emitted_over_measured": as_emitted / measured_cycles if measured_cycles else None,
            "verdict": ("FALSIFIED: the lower bound exceeds the measurement"
                        if cp.cycles > measured_cycles else
                        "consistent: the bound sits below the measurement, as a bound must"),
        }
    return report
