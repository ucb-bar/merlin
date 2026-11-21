"""``merlin-emit-dispatch-program`` — the outlined driver as a runtime dispatch table.

Given an outlined module (``outline_dispatches``), this reads the driver ``func @forward``
and emits a flat, serializable **dispatch program**: an ordered DAG of nodes the Merlin
runtime walks. Two node kinds:

- ``dispatch`` — invoke a compiled kernel symbol (``forward$kernel_<i>``) with input buffer
  ids, producing output buffer ids. Carries the root op and provenance.
- ``view`` — a cheap driver-side glue op (``tensor.extract_slice``/``expand_shape``/
  ``collapse_shape``/``cast``, scalar ``arith``) that derives an intermediate buffer from
  others. The runtime executes these directly; they are not compiled kernels.

Every value flowing through the driver becomes a **buffer** with a stable id, a shape and
a dtype. The program records the model arguments (inputs/weights by func-arg index), the
buffer table, the ordered nodes, and the result buffer ids. Because the ids are SSA
identities, the program is a DAG: each node's inputs are defined by an earlier node's
outputs or by a model argument — which :func:`verify_program` checks. This is the
target-agnostic command buffer that the Python simulator and the C runtime both consume,
and the unit a multicore scheduler partitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import prod
from typing import Any

from merlin.compiler.dialects.xdsl_context import HAS_XDSL
from merlin.compiler.ir.boundary import region_captures
from merlin.compiler.partition.outline import (
    OutlineError,
    OutlineResult,
    _external_declarations,
    outline_dispatches,
)

# ---- dispatch_program --------------------------------------------------------------------------

# NOTE: a `VIEW_OPS` tuple used to sit here, claiming to name "driver-side glue ops the runtime can
# evaluate directly". Nothing imported it, and it was WRONG in both directions against the only
# authority on the question -- `runtime.dispatch_runtime._eval_view`, which is the code that actually
# evaluates them. It listed tensor.extract_slice / insert_slice / cast / extract / from_elements, none
# of which _eval_view handles, and omitted linalg.fill / tensor.concat / tensor.splat, which it does.
# An unused constant that misstates a fact is worse than no constant: the next reader treats it as the
# spec. Deleted rather than corrected, because there is no second copy of this list to keep in sync --
# ask `_eval_view`.


@dataclass
class Buffer:
    id: str
    shape: list[int]
    dtype: str
    kind: str  # "arg" | "intermediate" | "const"
    arg_index: int | None = None  # set when kind == "arg"


@dataclass
class Node:
    kind: str  # "dispatch" | "view"
    op: str  # symbol for dispatch, op name for view
    inputs: list[str]  # ordered operand slots (including repeats), then additional region captures
    outputs: list[str]
    prov: dict[str, str] = field(default_factory=dict)
    #: How many regions the source op carries (0 for every flat op). A region-carrying node reads
    #: values its own operand list does NOT mention, so a consumer that reasons about liveness has to
    #: know the difference between "flat op" and "op with a body".
    regions: int = 0
    #: Buffer ids this node reads from INSIDE its regions (already folded into :attr:`inputs`).
    #: ``None`` means NOBODY COMPUTED THEM -- an older writer, or a hand-built program. That is a
    #: different statement from ``[]`` ("computed, the body captures nothing"), and the distinction is
    #: load-bearing: :func:`~.arena_plan.plan_arena` refuses to plan a region-carrying node whose
    #: capture set is unknown rather than assume the operand list is complete.
    captures: list[str] | None = None


@dataclass
class DispatchProgram:
    entry: str
    args: list[int]  # func-arg indices, in call order
    buffers: dict[str, Buffer]
    nodes: list[Node]
    results: list[str]

    @property
    def n_dispatches(self) -> int:
        return sum(1 for n in self.nodes if n.kind == "dispatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry,
            "args": list(self.args),
            "buffers": {k: asdict(v) for k, v in self.buffers.items()},
            "nodes": [asdict(n) for n in self.nodes],
            "results": list(self.results),
        }


def _shape_dtype(t) -> tuple[list[int], str]:
    from xdsl.dialects.builtin import TensorType

    if isinstance(t, TensorType):
        return [int(d) for d in t.get_shape()], str(t.element_type)
    return [], str(t)


def _nested_call_symbols(op) -> list[str]:
    """Kernel calls hiding inside ``op``'s regions, in body order (empty for a flat op).

    A ``func.call`` nested in a driver region is invisible to the top-level ``block.ops`` walk that
    pairs calls with the outlined dispatch table, so every LATER top-level call silently takes the
    WRONG entry: the third call in the IR gets the second kernel's symbol AND its ``prov.region_id``,
    and ``verify_program`` still returns ``[]`` because the buffer DAG stays self-consistent. There is
    no way to attribute such a program correctly, so it is refused rather than emitted.
    """
    found: list[str] = []
    for region in op.regions:
        for block in region.blocks:
            for top in block.ops:
                for sub in top.walk():
                    if sub.name == "func.call":
                        found.append(str(getattr(sub, "callee", sub.name)))
    return found


def _check_dispatch_definition(module, call, dispatch) -> None:
    """Join table intent to its actual typed definition, without execution proof."""
    from xdsl.dialects.builtin import IntegerAttr, StringAttr

    definitions = [op for op in module.body.block.ops if op.name == "func.func" and op.sym_name.data == dispatch.symbol]
    if len(definitions) != 1 or len(definitions[0].body.blocks) != 1:
        raise OutlineError(f"dispatch {dispatch.symbol!r} needs one actual defined function")
    function = definitions[0]
    signature = function.function_type
    if (
        tuple(value.type for value in call.operands) != signature.inputs.data
        or tuple(value.type for value in call.results) != signature.outputs.data
        or len(call.operands) != dispatch.n_operands
        or [str(value.type) for value in call.results] != dispatch.result_types
    ):
        raise OutlineError(f"dispatch {dispatch.symbol!r} disagrees with its actual ordered call/function ABI")
    for name, selected in (("merlin.placement", dispatch.placement), ("merlin.group", dispatch.group)):
        values = [table[name] for table in (function.attributes, function.properties) if name in table]
        if selected is None:
            agrees = not values
        elif name == "merlin.placement":
            agrees = (
                type(selected) is str
                and bool(selected.strip())
                and bool(values)
                and all(isinstance(value, StringAttr) and value.data == selected for value in values)
            )
        else:
            agrees = (
                type(selected) is int
                and selected >= 0
                and bool(values)
                and all(isinstance(value, IntegerAttr) and value.value.data == selected for value in values)
            )
        if not agrees:
            raise OutlineError(f"dispatch {dispatch.symbol!r} {name} disagrees with its actual function")
    call_placements = [
        table["merlin.placement"] for table in (call.attributes, call.properties) if "merlin.placement" in table
    ]
    if any(not isinstance(value, StringAttr) or value.data != dispatch.placement for value in call_placements):
        raise OutlineError(f"dispatch {dispatch.symbol!r} placement disagrees with its actual call")


def build_dispatch_program(outlined: OutlineResult, entry: str = "forward") -> DispatchProgram:
    """Flatten the outlined driver into a serializable dispatch program."""
    if not HAS_XDSL:
        raise OutlineError("xDSL is required to build a dispatch program")
    module = outlined.module
    declarations = _external_declarations(module, outlined.external_symbols)
    drivers = [
        op for op in module.walk() if op.name == "func.func" and op.body.blocks and "$kernel_" not in op.sym_name.data
    ]
    if not drivers:
        raise OutlineError("no driver func in outlined module")
    driver = next((d for d in drivers if d.sym_name.data == entry), drivers[0])
    block = driver.body.blocks[0]

    buffers: dict[str, Buffer] = {}
    ids: dict[int, str] = {}  # id(SSAValue) -> buffer id
    args: list[int] = []

    def bind(value, kind: str, arg_index: int | None = None) -> str:
        key = id(value)
        if key in ids:
            return ids[key]
        bid = f"b{len(buffers)}"
        shape, dtype = _shape_dtype(value.type)
        buffers[bid] = Buffer(id=bid, shape=shape, dtype=dtype, kind=kind, arg_index=arg_index)
        ids[key] = bid
        return bid

    # Model arguments (inputs + weights), in func-signature order.
    for i, a in enumerate(block.args):
        bind(a, "arg", arg_index=i)
        args.append(i)

    def resolve(values, op) -> list[str]:
        """Buffer ids for every ordered operand slot, including repeats, or raise.

        An unbound value used to be DROPPED (``if id(o) in ids``). A dropped input is a read the
        program does not record, which is the same silent liveness understatement region capture
        causes -- so say so instead."""
        out: list[str] = []
        for value in values:
            key = id(value)
            bid = ids.get(key)
            if bid is None:
                raise OutlineError(
                    f"driver op {op.name!r} reads a value with no bound buffer ({value.type}). "
                    "Dropping it (the retired behavior) records a read the program never performs, "
                    "which understates the buffer's live range for every downstream consumer."
                )
            out.append(bid)
        return out

    nodes: list[Node] = []
    disp_iter = iter(outlined.dispatches)
    n_calls = 0
    for op in block.ops:
        if op.name == "func.return":
            continue
        if op.name == "func.call":
            callee = op.callee.string_value()
            if callee in declarations:
                # Explicit catalog calls are nodes in their original position, without consuming
                # an outlined-kernel table entry or claiming the declaration is an implementation.
                nodes.append(
                    Node(
                        kind="dispatch",
                        op=callee,
                        inputs=resolve(op.operands, op),
                        outputs=[bind(r, "intermediate") for r in op.results],
                    )
                )
                continue
            d = next(disp_iter, None)
            if d is None:
                raise OutlineError(
                    f"dispatch table exhausted at driver call #{n_calls + 1}: the driver makes more "
                    f"kernel calls than the outline result lists dispatches "
                    f"({len(outlined.dispatches)}). Pairing them positionally past this point would "
                    "attribute a call to the wrong kernel symbol and the wrong provenance."
                )
            if callee != d.symbol:
                raise OutlineError(f"dispatch table symbol {d.symbol!r} disagrees with driver call {callee!r}")
            if type(d.index) is not int or d.index != n_calls:
                raise OutlineError("dispatch table indices must enumerate actual call order exactly")
            _check_dispatch_definition(module, op, d)
            n_calls += 1
            in_ids = resolve(op.operands, op)
            out_ids = [bind(r, "intermediate") for r in op.results]
            nodes.append(
                Node(
                    kind="dispatch",
                    op=d.symbol,
                    inputs=in_ids,
                    outputs=out_ids,
                    prov=d.prov,
                    regions=len(op.regions),
                    captures=[],
                )
            )
            continue
        # a driver-side glue / view op
        nested = _nested_call_symbols(op)
        if nested:
            raise OutlineError(
                f"driver op {op.name!r} carries a region containing kernel call(s) {nested}: the "
                "top-level walk that pairs driver calls with the dispatch table cannot see them, so "
                "every later call would silently take the wrong dispatch entry (wrong symbol AND "
                "wrong prov.region_id) while verify_program still reports a valid DAG. Hoist the "
                "call out of the region before building a dispatch program."
            )
        captured = region_captures(op) if op.regions else []
        kind = "const" if op.name == "arith.constant" else "intermediate"
        in_ids = resolve(op.operands, op)
        capture_ids = resolve(captured, op)
        in_ids.extend(bid for bid in capture_ids if bid not in in_ids)
        out_ids = [bind(r, kind) for r in op.results]
        nodes.append(
            Node(kind="view", op=op.name, inputs=in_ids, outputs=out_ids, regions=len(op.regions), captures=capture_ids)
        )

    leftover = [d.symbol for d in disp_iter]
    if leftover:
        raise OutlineError(
            f"dispatch table desynchronised: {len(leftover)} outlined dispatch(es) were never "
            f"matched to a driver call ({leftover}); the driver made {n_calls} top-level call(s) for "
            f"{len(outlined.dispatches)} dispatches. Every call after the first unmatched one already "
            "carries another kernel's symbol and provenance, and the buffer DAG stays valid, so this "
            "is the only place the mismatch is visible."
        )

    ret = next(op for op in block.ops if op.name == "func.return")
    results = [ids[id(o)] for o in ret.operands]
    return DispatchProgram(entry=driver.sym_name.data, args=args, buffers=buffers, nodes=nodes, results=results)


def verify_program(prog: DispatchProgram) -> list[str]:
    """Check the program is a well-formed DAG over its buffers; return problems."""
    problems: list[str] = []
    defined: set[str] = {b.id for b in prog.buffers.values() if b.kind in ("arg", "const")}
    for n, node in enumerate(prog.nodes):
        for bid in node.inputs:
            if bid not in prog.buffers:
                problems.append(f"node {n} ({node.op}) reads unknown buffer {bid}")
            elif bid not in defined:
                problems.append(f"node {n} ({node.op}) reads buffer {bid} before it is produced")
        for bid in node.outputs:
            defined.add(bid)
    for bid in prog.results:
        if bid not in defined:
            problems.append(f"result buffer {bid} is never produced")
    return problems


def prune_dead_nodes(prog: DispatchProgram) -> DispatchProgram:
    """Drop view nodes whose results never reach a dispatch input or a program result.

    The outliner clones each kernel's accumulator init (``tensor.empty``/``arith.constant``
    /``linalg.fill``) into the kernel, leaving dead copies in the driver. Those become
    dead view nodes here; eliminating them yields the minimal dispatch program. Dispatch
    nodes are always kept (they are the compute).
    """
    needed: set[str] = set(prog.results)
    for node in prog.nodes:
        if node.kind == "dispatch":
            needed.update(node.inputs)

    keep = [False] * len(prog.nodes)
    for i in range(len(prog.nodes) - 1, -1, -1):
        node = prog.nodes[i]
        if node.kind == "dispatch" or any(o in needed for o in node.outputs):
            keep[i] = True
            needed.update(node.inputs)

    nodes = [n for n, k in zip(prog.nodes, keep) if k]
    live: set[str] = set(prog.results)
    for node in nodes:
        live.update(node.inputs)
        live.update(node.outputs)
    buffers = {bid: b for bid, b in prog.buffers.items() if bid in live}
    return DispatchProgram(
        entry=prog.entry, args=list(prog.args), buffers=buffers, nodes=nodes, results=list(prog.results)
    )


def slice_program(prog: DispatchProgram, region_ids, *, entry_suffix: str = "") -> DispatchProgram:
    """Slice a whole-model dispatch program down to a SECTION: the dispatch nodes whose
    ``prov.region_id`` is in ``region_ids`` plus the interior view-glue between them.

    This is the runtime side of "compile the whole model, profile only a section". The result is a
    valid, standalone ``DispatchProgram``: cross-boundary inputs (buffers a kept node reads but no kept
    node produces) are reclassified to ``kind="arg"`` — they are fed from the region-boundary tensors
    (``region_goldens.npz``); the slice's ``results`` are the buffers that leave it (consumed outside
    the slice, an original program result, or a terminal section output). ``plan_arena`` re-plans it
    as-is → a smaller arena. Selecting several region_ids at once yields ONE combined sub-program;
    call once per region for separate binaries. Generalizes :func:`prune_dead_nodes` (backward
    reachability), keyed on region provenance instead of program results. Raises if nothing matches.
    """
    want = set(region_ids)
    keep: set[int] = {
        i for i, n in enumerate(prog.nodes) if n.kind == "dispatch" and n.prov.get("prov.region_id") in want
    }
    if not keep:
        raise ValueError(f"slice_program: no dispatch nodes match region_ids {sorted(want)}")

    # Interior view-glue: a view node is kept when EVERY consumer of its outputs is already kept (it is
    # internal to the slice). Fixpoint so a chain of views collapses in. A view feeding outside the
    # slice is NOT pulled in — its output becomes a boundary result/input instead.
    consumers: dict[str, list[int]] = {}
    for i, n in enumerate(prog.nodes):
        for b in n.inputs:
            consumers.setdefault(b, []).append(i)
    changed = True
    while changed:
        changed = False
        for i, n in enumerate(prog.nodes):
            if i in keep or n.kind != "view":
                continue
            cons = [c for b in n.outputs for c in consumers.get(b, [])]
            if cons and all(c in keep for c in cons):
                keep.add(i)
                changed = True

    kept = [prog.nodes[i] for i in sorted(keep)]
    produced = {b for n in kept for b in n.outputs}
    outside_consumed = {b for i, n in enumerate(prog.nodes) if i not in keep for b in n.inputs}
    consumed_anywhere = {b for n in prog.nodes for b in n.inputs}
    prog_results = set(prog.results)

    # Slice results: a kept output that leaves the slice (consumed outside / a program result) or is a
    # terminal section output (produced, consumed nowhere) — so the section's product is always exposed.
    results: list[str] = []
    for n in kept:
        for b in n.outputs:
            leaves = b in outside_consumed or b in prog_results
            terminal = n.kind == "dispatch" and b not in consumed_anywhere and b not in prog_results
            if (leaves or terminal) and b not in results:
                results.append(b)

    # Rebuild the buffer table: internal buffers keep their kind; a needed-but-not-produced input
    # becomes a boundary arg (a const travels with the slice).
    buffers: dict[str, Buffer] = {}

    def _clone(bid: str, *, kind: str | None = None, arg_index: int | None = None) -> None:
        s = prog.buffers[bid]
        buffers[bid] = Buffer(id=s.id, shape=list(s.shape), dtype=s.dtype, kind=kind or s.kind, arg_index=arg_index)

    args: list[int] = []
    for bid in dict.fromkeys(b for n in kept for b in n.inputs):
        if bid in produced:
            continue
        if prog.buffers[bid].kind == "const":
            _clone(bid)  # consts are emitted with the slice
        else:
            _clone(bid, kind="arg", arg_index=len(args))  # boundary input -> a fresh slice arg
            args.append(buffers[bid].arg_index)
    for bid in produced:
        _clone(bid)
    for bid in results:  # ensure every result buffer is present
        if bid not in buffers:
            _clone(bid)

    sliced = DispatchProgram(entry=prog.entry + entry_suffix, args=args, buffers=buffers, nodes=kept, results=results)
    problems = verify_program(sliced)
    if problems:
        raise ValueError(f"slice_program produced an invalid DAG: {problems}")
    return sliced


def lower_model_to_dispatch_program(
    module, forward: str | None = None, prune: bool = True, *, external_symbols=()
) -> tuple[OutlineResult, DispatchProgram]:
    """Convenience: outline then flatten, verifying the resulting program."""
    outlined = outline_dispatches(module, forward=forward, external_symbols=external_symbols)
    prog = build_dispatch_program(outlined, entry=forward or "forward")
    if prune:
        prog = prune_dead_nodes(prog)
    problems = verify_program(prog)
    if problems:
        raise OutlineError("invalid dispatch program: " + "; ".join(problems[:5]))
    return outlined, prog


# ---- schedule_dispatch -------------------------------------------------------------------------
# ``merlin-partition-dispatches`` — schedule the dispatch DAG across harts.
#
# The dispatch program (``dispatch_program.py``) is a pure-dataflow DAG: every buffer is
# written by exactly one node, so any dependency-respecting execution order yields identical
# results. That makes multicore scheduling a graph problem, not a correctness risk: place
# each node on a hart and in an order such that no node runs before a node that produces one
# of its inputs, and independent nodes run concurrently.
#
# This produces a **level-synchronous schedule** — nodes grouped into dependency levels
# (ASAP longest-path), a barrier between levels, and within each level the nodes balanced
# across ``n_harts`` by longest-processing-time-first. Because a data dependency u→v forces
# ``level(v) > level(u)``, no two nodes in the same level depend on each other, so running a
# whole level in parallel is always safe (`validate` proves it). The C / spike runtime walks
# this schedule: each hart runs its slice of a level, then all harts sync at the barrier.
#
# Cost model (for balancing + the makespan estimate): a matmul costs M·N·K, a batched
# matmul B·M·N·K, everything else its output element count. The reported speedup is the
# serial cost over the level-synchronous makespan — the parallelism actually available in
# the model (intra-layer: independent q/k/v projections, attention heads, elementwise).


@dataclass
class Schedule:
    n_harts: int
    levels: list[list[int]]  # node indices per dependency level (barriers between)
    hart_of: dict[int, int]  # node index -> hart
    cost_of: dict[int, int]  # node index -> cost
    serial_cost: int  # sum of all node costs (1 hart, no parallelism)
    critical_path_cost: int  # longest dependency chain by cost (ideal lower bound)
    makespan: int  # level-synchronous cost with n_harts

    @property
    def speedup(self) -> float:
        return self.serial_cost / max(1, self.makespan)

    @property
    def depth(self) -> int:
        return len(self.levels)

    @property
    def max_width(self) -> int:
        return max((len(lv) for lv in self.levels), default=0)

    def hart_loads(self) -> list[int]:
        loads = [0] * self.n_harts
        for n, h in self.hart_of.items():
            loads[h] += self.cost_of[n]
        return loads


def _producers(prog) -> dict[str, int]:
    """buffer id -> index of the node that produces it (single-writer dataflow)."""
    out: dict[str, int] = {}
    for i, node in enumerate(prog.nodes):
        for b in node.outputs:
            out[b] = i
    return out


def dependencies(prog) -> list[set[int]]:
    """For each node, the set of predecessor node indices (producers of its inputs)."""
    prod_of = _producers(prog)
    deps: list[set[int]] = []
    for node in prog.nodes:
        deps.append({prod_of[b] for b in node.inputs if b in prod_of})
    return deps


def node_cost(prog, idx: int) -> int:
    node = prog.nodes[idx]
    out_elems = 1
    if node.outputs:
        out_elems = max(prod(prog.buffers[b].shape) or 1 for b in node.outputs)
    op = node.prov.get("prov.op", "") if node.kind == "dispatch" else ""
    if op in ("matmul", "batch_matmul"):
        # contraction dim K from the first input's trailing dim
        k = 1
        if node.inputs:
            shp = prog.buffers[node.inputs[0]].shape
            k = shp[-1] if shp else 1
        return max(1, out_elems * k)
    return max(1, out_elems)


def asap_levels(prog, deps: list[set[int]]) -> list[int]:
    """Longest-path level of each node (nodes are already in a topological order)."""
    level = [0] * len(prog.nodes)
    for i in range(len(prog.nodes)):
        if deps[i]:
            level[i] = 1 + max(level[p] for p in deps[i])
    return level


def schedule(prog, n_harts: int = 4) -> Schedule:
    """Level-synchronous, load-balanced schedule of the dispatch DAG over ``n_harts``."""
    deps = dependencies(prog)
    level = asap_levels(prog, deps)
    cost = {i: node_cost(prog, i) for i in range(len(prog.nodes))}

    levels: list[list[int]] = [[] for _ in range(max(level, default=-1) + 1)]
    for i, lv in enumerate(level):
        levels[lv].append(i)

    # within each level: longest-processing-time-first onto the least-loaded hart
    hart_of: dict[int, int] = {}
    makespan = 0
    for lv in levels:
        loads = [0] * n_harts
        for n in sorted(lv, key=lambda x: cost[x], reverse=True):
            h = min(range(n_harts), key=lambda x: loads[x])
            hart_of[n] = h
            loads[h] += cost[n]
        makespan += max(loads) if loads else 0

    serial = sum(cost.values())
    # critical path by cost (ideal lower bound regardless of hart count)
    cp = [0] * len(prog.nodes)
    for i in range(len(prog.nodes)):
        cp[i] = cost[i] + (max((cp[p] for p in deps[i]), default=0))
    critical = max(cp, default=0)

    return Schedule(
        n_harts=n_harts,
        levels=levels,
        hart_of=hart_of,
        cost_of=cost,
        serial_cost=serial,
        critical_path_cost=critical,
        makespan=makespan,
    )


def validate(prog, sched: Schedule) -> list[str]:
    """Prove the schedule is dependency-safe: every edge crosses a barrier upward.

    For each dependency u→v, ``level(u) < level(v)`` (so u runs, then a barrier, then v).
    Equivalently no two nodes in one level depend on each other — making whole-level
    parallel execution correct on a single-writer dataflow program.
    """
    problems: list[str] = []
    level_of: dict[int, int] = {}
    for lv, nodes in enumerate(sched.levels):
        for n in nodes:
            level_of[n] = lv
    deps = dependencies(prog)
    for v, preds in enumerate(deps):
        for u in preds:
            if level_of.get(u, -1) >= level_of.get(v, -1):
                problems.append(
                    f"node {v} (level {level_of.get(v)}) depends on node {u} "
                    f"(level {level_of.get(u)}) — not strictly earlier"
                )
    assigned = set(sched.hart_of)
    if assigned != set(range(len(prog.nodes))):
        problems.append("not every node is assigned to a hart")
    return problems


@dataclass
class PartitionResult:
    schedule: Schedule
    stats: dict[str, Any] = field(default_factory=dict)


def partition_dispatches(prog, n_harts: int = 4) -> PartitionResult:
    """Schedule + validate a dispatch program for multicore; raise on an invalid schedule."""
    if not HAS_XDSL:
        raise RuntimeError("xDSL is required")
    sched = schedule(prog, n_harts=n_harts)
    problems = validate(prog, sched)
    if problems:
        raise RuntimeError("invalid schedule: " + "; ".join(problems[:5]))
    stats = {
        "n_harts": n_harts,
        "nodes": len(prog.nodes),
        "depth": sched.depth,
        "max_width": sched.max_width,
        "serial_cost": sched.serial_cost,
        "critical_path_cost": sched.critical_path_cost,
        "makespan": sched.makespan,
        "speedup": round(sched.speedup, 2),
        "hart_loads": sched.hart_loads(),
    }
    return PartitionResult(schedule=sched, stats=stats)


def emit_schedule_c(prog, sched: Schedule, name: str = "MERLIN_SCHEDULE") -> str:
    """Emit the partitioned schedule as a C header the multicore runtime consumes.

    One ``merlin_dispatch_t`` row per dispatch node, in (level, hart) order: the kernel
    symbol, its dependency level, the hart it runs on, and the buffer ids it reads/writes.
    A level-synchronous executor (e.g. the Zephyr ``merlin_mt_rvv_dispatch`` thread pool)
    runs all rows of one level across harts, barriers, then advances — exactly the
    structure validated by :func:`validate`.
    """
    buf_index = {bid: i for i, bid in enumerate(sorted(prog.buffers))}
    rows = []
    order = sorted(
        range(len(prog.nodes)),
        key=lambda n: (next(l for l, ns in enumerate(sched.levels) if n in ns), sched.hart_of[n]),
    )
    for n in order:
        node = prog.nodes[n]
        level = next(l for l, ns in enumerate(sched.levels) if n in ns)
        ins = node.inputs + [""] * (4 - len(node.inputs))
        in_idx = ", ".join(str(buf_index.get(b, -1)) for b in ins[:4])
        out_idx = buf_index.get(node.outputs[0], -1) if node.outputs else -1
        rows.append(f'  {{ "{node.op}", {level}, {sched.hart_of[n]}, {{{in_idx}}}, {len(node.inputs)}, {out_idx} }},')
    lines = [
        "/* Generated by merlin.compiler.partition.dispatch_program. */",
        "#ifndef MERLIN_SCHEDULE_H",
        "#define MERLIN_SCHEDULE_H",
        "typedef struct { const char *kernel; int level; int hart;",
        "                 int in_buf[4]; int n_in; int out_buf; } merlin_dispatch_t;",
        f"#define {name}_N {len(rows)}",
        f"#define {name}_LEVELS {sched.depth}",
        f"#define {name}_HARTS {sched.n_harts}",
        f"static const merlin_dispatch_t {name}[{name}_N] = {{",
        *rows,
        "};",
        "#endif",
    ]
    return "\n".join(lines) + "\n"
