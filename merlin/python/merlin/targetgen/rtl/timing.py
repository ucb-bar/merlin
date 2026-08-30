"""RTL-derived TIMING facts — how many clocked stages a module's outputs sit behind its inputs.

Merlin's facts said nothing about TIME. Geometry, capacities and the decode table were all derived
from the RTL, but "how long does this unit take" was not a fact any extractor produced — so a target
whose ISA schedules statically (no interlocks: the *program* carries the delays) could not be
compiled correctly from facts alone, and a scheduler had nothing to schedule against.

The number is DERIVED, never declared. A vendor performance model that ships a per-op latency table
is a hand-written artifact of theirs: it may disagree with their own RTL, it will not exist for the
next target, and inheriting it would make merlin's facts only as true as somebody else's model. The
RTL knows the answer structurally — a pipeline stage IS a register on the path — so we count them.

METHOD. For each ``hw.module``, walk the use-def graph backward from every operand of ``hw.output``
to the module's input ports, counting ``seq.firreg`` crossings, and take the longest path. That is
the module's pipeline depth in cycles. The walk is target-agnostic: it names no module, no unit and
no target, and runs over whatever modules the design happens to contain.

WHAT IT CANNOT ANSWER, said out loud. The walk measures FEED-FORWARD depth. A sequenced unit — an
FSM, a counter-driven sequencer, a queue — reaches its outputs through a feedback loop, and "the
longest path" through a cycle is not a finite number. Those modules report ``pipeline_depth: None``:
their latency is a function of state and operands (tile rows, byte counts), not of wiring depth, and
recovering it needs the sequencer's own limits or a measurement. Reporting the acyclic subset's
maximum AS the depth would be the flattering answer to a question this method cannot reach, so a
module with any cyclic output records ``partial_depth`` under its own name and leaves
``pipeline_depth`` UNKNOWN. Two differently-derived numbers must not share one field.
"""
from __future__ import annotations

from typing import Any

#: Sentinel: this value's cone contains a feedback path, so no finite depth exists for it.
CYCLIC = object()

#: The clocked element. A register on a path IS a pipeline stage -- that is the whole derivation.
_REG_OPS = ("seq.firreg", "seq.compreg")
#: Constants terminate a path without contributing depth (they are not driven by an input).
_LEAF_OPS = ("hw.constant",)


def _key(value) -> int:
    return id(value)


def value_depth(graph, root) -> int | object:
    """Longest register count from ``root`` back to the module's inputs, or :data:`CYCLIC`.

    Iterative (an explicit stack, not recursion): a wide combinational cone in a real design is
    deeper than the interpreter's frame limit, and a RecursionError here would look like a module
    with no answer rather than a walker that gave up.
    """
    done: dict[int, Any] = {}
    on_stack: set[int] = set()
    stack: list[tuple[Any, bool]] = [(root, False)]

    while stack:
        value, expanded = stack.pop()
        k = _key(value)
        if expanded:
            on_stack.discard(k)
            op = graph.defining_op(value)
            name = op.op_name.data if op is not None else ""
            if name in _REG_OPS:
                # A register's DATA operand is operand 0; clock/reset carry no data depth.
                sub = done.get(_key(op.operands[0])) if op.operands else 0
                done[k] = CYCLIC if sub is CYCLIC else (sub or 0) + 1
            else:
                best = 0
                for operand in (op.operands if op is not None else ()):
                    sub = done.get(_key(operand), 0)
                    if sub is CYCLIC:
                        best = CYCLIC
                        break
                    best = max(best, sub)
                done[k] = best
            continue

        if k in done:
            continue
        if k in on_stack:
            # A back edge: this value's cone re-enters itself, so it is sequenced, not pipelined.
            done[k] = CYCLIC
            continue

        op = graph.defining_op(value)
        # A module input port (block argument) is depth 0 -- it is where the walk is trying to get.
        if op is None or graph.is_block_arg(value):
            done[k] = 0
            continue
        name = op.op_name.data
        if name in _LEAF_OPS:
            done[k] = 0
            continue

        on_stack.add(k)
        stack.append((value, True))
        operands = op.operands[:1] if name in _REG_OPS else op.operands
        for operand in operands:
            if _key(operand) not in done:
                stack.append((operand, False))

    return done.get(_key(root), 0)


def module_timing(graph, module) -> dict[str, Any]:
    """Pipeline depth for one ``hw.module``, with the evidence that produced it.

    ``pipeline_depth`` is set ONLY when every output is acyclic. When some output is sequenced the
    depth is UNKNOWN and the acyclic maximum is reported as ``partial_depth`` -- a different name,
    because it answers a different question and must never be read as the module's latency.
    """
    outputs = graph.ops("hw.output", within=module)
    registers = len(graph.ops("seq.firreg", within=module))
    rec: dict[str, Any] = {
        "module": module.name,
        "registers": registers,
        "source": "mlc_hw_graph_walk",
    }
    driven = [v for out in outputs for v in out.operands]
    if not driven:
        rec.update(pipeline_depth=None, partial_depth=None, n_outputs=0, n_cyclic=0,
                   evidence=f"module {module.name} drives no hw.output operand: nothing to walk")
        return rec

    depths = [value_depth(graph, v) for v in driven]
    acyclic = [d for d in depths if d is not CYCLIC]
    n_cyclic = len(depths) - len(acyclic)
    rec["n_outputs"] = len(depths)
    rec["n_cyclic"] = n_cyclic

    if n_cyclic == 0:
        depth = max(acyclic)
        rec.update(pipeline_depth=depth, partial_depth=None,
                   evidence=f"longest seq.firreg chain from an input port to any of "
                            f"{len(depths)} hw.output operands in module {module.name} "
                            f"({registers} registers, no feedback on any output) = {depth}")
    else:
        rec.update(pipeline_depth=None,
                   partial_depth=(max(acyclic) if acyclic else None),
                   evidence=f"{n_cyclic} of {len(depths)} hw.output operands of module "
                            f"{module.name} are reached through feedback: this unit is SEQUENCED, "
                            f"so no finite wiring depth is its latency (needs the sequencer's own "
                            f"limits or a measurement)")
    return rec


def discovered_timing(target: str) -> list[dict[str, Any]] | None:
    """Per-module pipeline depth for ``target``, or None when the RTL is not reachable.

    None means UNKNOWN -- mlc absent, or no HW dialect for this target. It never means "this design
    has no timing", and it is never an empty list standing in for a design nobody looked at.
    """
    from . import mlc_bridge

    available, _why = mlc_bridge.mlc_available()
    if not available:
        return None
    hw = mlc_bridge.core_hw_mlir(target)
    if hw is None:
        return None
    from mlc.discover import irgraph

    graph = irgraph.load_hw_graph(hw, circt_opt=mlc_bridge.circt_opt_bin())
    return [module_timing(graph, m) for m in graph.modules.values()]
