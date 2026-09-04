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
from typing import Any

from .._common import HAS_XDSL
from .outline import OutlineError, OutlineResult, outline_dispatches

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
    kind: str                       # "arg" | "intermediate" | "const"
    arg_index: int | None = None    # set when kind == "arg"


@dataclass
class Node:
    kind: str                       # "dispatch" | "view"
    op: str                         # symbol for dispatch, op name for view
    inputs: list[str]
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
    args: list[int]                 # func-arg indices, in call order
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


def _region_captures(op) -> list:
    """Values ``op``'s REGIONS read from the enclosing scope -- the operands ``op.operands`` omits.

    ``op.operands`` is the op's own signature and nothing more: for ``scf.for`` that is
    ``lb``/``ub``/``step``/``iter_args``. A tensor the loop BODY reads out of the enclosing scope is
    not an operand of the loop at all, so a node built from ``op.operands`` alone understates what the
    node reads -- and a memory planner fed that node computes a live range that ends before the last
    read. It then hands those bytes to another buffer while the loop is still reading them: wrong
    numbers, no diagnostic, and only on the input sizes that happen to collide.

    A value is captured when it is used anywhere inside the regions (at any nesting depth) and is NOT
    defined inside them -- i.e. it is neither the result of an inner op nor a block argument of any
    inner block (a loop induction variable and an ``iter_args`` body argument are both block args, so
    both are correctly internal). Same rule the outliner already applies in ``_free_values``; stated
    once more here because this is a different question about the same structure.
    """
    internal: set[int] = set()
    inner_ops: list = []
    for region in op.regions:
        for block in region.blocks:
            for arg in block.args:
                internal.add(id(arg))
            for top in block.ops:
                for sub in top.walk():
                    inner_ops.append(sub)
                    for res in sub.results:
                        internal.add(id(res))
                    for sub_region in sub.regions:
                        for sub_block in sub_region.blocks:
                            for sub_arg in sub_block.args:
                                internal.add(id(sub_arg))
    captured: list = []
    seen: set[int] = set()
    for sub in inner_ops:
        for value in sub.operands:
            key = id(value)
            if key in internal or key in seen:
                continue
            seen.add(key)
            captured.append(value)
    return captured


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


def build_dispatch_program(outlined: OutlineResult, entry: str = "forward"
                           ) -> DispatchProgram:
    """Flatten the outlined driver into a serializable dispatch program."""
    if not HAS_XDSL:
        raise OutlineError("xDSL is required to build a dispatch program")
    module = outlined.module
    drivers = [op for op in module.walk()
               if op.name == "func.func" and "$kernel_" not in op.sym_name.data]
    if not drivers:
        raise OutlineError("no driver func in outlined module")
    driver = next((d for d in drivers if d.sym_name.data == entry), drivers[0])
    block = driver.body.blocks[0]

    buffers: dict[str, Buffer] = {}
    ids: dict[int, str] = {}          # id(SSAValue) -> buffer id
    args: list[int] = []

    def bind(value, kind: str, arg_index: int | None = None) -> str:
        key = id(value)
        if key in ids:
            return ids[key]
        bid = f"b{len(buffers)}"
        shape, dtype = _shape_dtype(value.type)
        buffers[bid] = Buffer(id=bid, shape=shape, dtype=dtype, kind=kind,
                              arg_index=arg_index)
        ids[key] = bid
        return bid

    # Model arguments (inputs + weights), in func-signature order.
    for i, a in enumerate(block.args):
        bind(a, "arg", arg_index=i)
        args.append(i)

    def resolve(values, op) -> list[str]:
        """Buffer ids for ``values``, order-preserving and de-duplicated, or raise.

        An unbound value used to be DROPPED (``if id(o) in ids``). A dropped input is a read the
        program does not record, which is the same silent liveness understatement region capture
        causes -- so say so instead."""
        out: list[str] = []
        seen: set[int] = set()
        for value in values:
            key = id(value)
            if key in seen:
                continue
            seen.add(key)
            bid = ids.get(key)
            if bid is None:
                raise OutlineError(
                    f"driver op {op.name!r} reads a value with no bound buffer ({value.type}). "
                    "Dropping it (the retired behavior) records a read the program never performs, "
                    "which understates the buffer's live range for every downstream consumer.")
            out.append(bid)
        return out

    nodes: list[Node] = []
    disp_iter = iter(outlined.dispatches)
    n_calls = 0
    for op in block.ops:
        if op.name == "func.return":
            continue
        if op.name == "func.call":
            d = next(disp_iter, None)
            if d is None:
                raise OutlineError(
                    f"dispatch table exhausted at driver call #{n_calls + 1}: the driver makes more "
                    f"kernel calls than the outline result lists dispatches "
                    f"({len(outlined.dispatches)}). Pairing them positionally past this point would "
                    "attribute a call to the wrong kernel symbol and the wrong provenance.")
            n_calls += 1
            in_ids = resolve(op.operands, op)
            out_ids = [bind(r, "intermediate") for r in op.results]
            nodes.append(Node(kind="dispatch", op=d.symbol, inputs=in_ids,
                              outputs=out_ids, prov=d.prov, regions=len(op.regions), captures=[]))
            continue
        # a driver-side glue / view op
        nested = _nested_call_symbols(op)
        if nested:
            raise OutlineError(
                f"driver op {op.name!r} carries a region containing kernel call(s) {nested}: the "
                "top-level walk that pairs driver calls with the dispatch table cannot see them, so "
                "every later call would silently take the wrong dispatch entry (wrong symbol AND "
                "wrong prov.region_id) while verify_program still reports a valid DAG. Hoist the "
                "call out of the region before building a dispatch program.")
        captured = _region_captures(op) if op.regions else []
        kind = "const" if op.name == "arith.constant" else "intermediate"
        in_ids = resolve(list(op.operands) + captured, op)
        capture_ids = [bid for bid in resolve(captured, op)] if captured else []
        out_ids = [bind(r, kind) for r in op.results]
        nodes.append(Node(kind="view", op=op.name, inputs=in_ids, outputs=out_ids,
                          regions=len(op.regions), captures=capture_ids))

    leftover = [d.symbol for d in disp_iter]
    if leftover:
        raise OutlineError(
            f"dispatch table desynchronised: {len(leftover)} outlined dispatch(es) were never "
            f"matched to a driver call ({leftover}); the driver made {n_calls} top-level call(s) for "
            f"{len(outlined.dispatches)} dispatches. Every call after the first unmatched one already "
            "carries another kernel's symbol and provenance, and the buffer DAG stays valid, so this "
            "is the only place the mismatch is visible.")

    ret = next(op for op in block.ops if op.name == "func.return")
    results = [ids[id(o)] for o in ret.operands]
    return DispatchProgram(entry=driver.sym_name.data, args=args, buffers=buffers,
                           nodes=nodes, results=results)


def verify_program(prog: DispatchProgram) -> list[str]:
    """Check the program is a well-formed DAG over its buffers; return problems."""
    problems: list[str] = []
    defined: set[str] = {b.id for b in prog.buffers.values() if b.kind in ("arg", "const")}
    for n, node in enumerate(prog.nodes):
        for bid in node.inputs:
            if bid not in prog.buffers:
                problems.append(f"node {n} ({node.op}) reads unknown buffer {bid}")
            elif bid not in defined:
                problems.append(
                    f"node {n} ({node.op}) reads buffer {bid} before it is produced")
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
    return DispatchProgram(entry=prog.entry, args=list(prog.args), buffers=buffers,
                           nodes=nodes, results=list(prog.results))


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
    keep: set[int] = {i for i, n in enumerate(prog.nodes)
                      if n.kind == "dispatch" and n.prov.get("prov.region_id") in want}
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
        buffers[bid] = Buffer(id=s.id, shape=list(s.shape), dtype=s.dtype,
                              kind=kind or s.kind, arg_index=arg_index)

    args: list[int] = []
    for bid in dict.fromkeys(b for n in kept for b in n.inputs):
        if bid in produced:
            continue
        if prog.buffers[bid].kind == "const":
            _clone(bid)                                  # consts are emitted with the slice
        else:
            _clone(bid, kind="arg", arg_index=len(args))  # boundary input -> a fresh slice arg
            args.append(buffers[bid].arg_index)
    for bid in produced:
        _clone(bid)
    for bid in results:                                  # ensure every result buffer is present
        if bid not in buffers:
            _clone(bid)

    sliced = DispatchProgram(entry=prog.entry + entry_suffix, args=args,
                             buffers=buffers, nodes=kept, results=results)
    problems = verify_program(sliced)
    if problems:
        raise ValueError(f"slice_program produced an invalid DAG: {problems}")
    return sliced


def lower_model_to_dispatch_program(module, forward: str | None = None,
                                    prune: bool = True
                                    ) -> tuple[OutlineResult, DispatchProgram]:
    """Convenience: outline then flatten, verifying the resulting program."""
    outlined = outline_dispatches(module, forward=forward)
    prog = build_dispatch_program(outlined, entry=forward or "forward")
    if prune:
        prog = prune_dead_nodes(prog)
    problems = verify_program(prog)
    if problems:
        raise OutlineError("invalid dispatch program: " + "; ".join(problems[:5]))
    return outlined, prog
