"""Exact finite transfer analysis for lexical SCF completion-token ownership.

Static stage footprints form the abstract state. Repeated states close counted
loops without enumerating trips or guessing device latency. These checks cover
declared RAW/WAR/WAW lifetimes only; target completion callbacks still need the
selected target's independent source/effect contract.
"""
from __future__ import annotations

import json

from xdsl.dialects import arith, builtin, func, scf

CARRIED_EVENTS = "merlin.scf_commands.carried_events.v1"


def loop_event_names(op):
    marker = op.attributes.get(CARRIED_EVENTS)
    if marker is None:
        if op.iter_args or op.results or len(op.body.block.args) != 1:
            raise ValueError("carried completion requires its explicit lexical loop bindings")
        return (), ()
    if type(marker) is not builtin.StringAttr:
        raise ValueError("malformed lexical completion loop bindings")
    data = json.loads(marker.data)
    if type(data) is not dict or set(data) != {"inputs", "results"}:
        raise ValueError("unknown lexical completion loop binding fields")
    inputs, results = data["inputs"], data["results"]
    if (type(inputs) is not list or type(results) is not list or not inputs
            or len(inputs) != len(results) or len(inputs) != len(op.iter_args)
            or len(op.results) != len(inputs) or len(op.body.block.args) != len(inputs) + 1
            or any(type(n) is not str or not n for n in (*inputs, *results))
            or len(set(inputs)) != len(inputs) or len(set(results)) != len(results)
            or any(v.type.name != "runtime.event" for v in
                   (*op.iter_args, *op.results, *op.body.block.args[1:]))):
        raise ValueError("completion loop input/result types or arities differ")
    return tuple(inputs), tuple(results)


def verify_carried_events(module, *, max_operations, allow_persistent_state):
    from . import scf_commands as C

    if type(module.attributes.get(CARRIED_EVENTS)) is not builtin.UnitAttr:
        raise ValueError("loop-carried completion requires the explicit versioned program marker")
    entry = C._entry(module)
    declarations = {op.sym_name.data: op for op in module.body.block.ops if not op.body.blocks}
    work = 0
    summaries = 0

    def body(block, incoming=(), input_names=(), *, yielding=False):
        nonlocal work, summaries
        values = dict(zip(block.args[1:], incoming, strict=True)) if yielding else {}
        names = dict(zip(block.args[1:], input_names, strict=True)) if yielding else {}
        pending = {v: state for v, state in values.items() if state is not None}
        # None is completed; an empty tuple is still an outstanding event with
        # no declared private rows. Completion can never be inferred from it.
        for op in block.ops:
            work += 1
            if work > max_operations:
                raise ValueError("completion-state transfer proof exceeds its explicit operation budget")
            if C.PERSISTENT_STATE in op.attributes and type(op) is not func.CallOp:
                raise ValueError("persistent state declarations belong to actual command calls only")
            if type(op) is scf.ForOp:
                local_names, result_names = loop_event_names(op)
                low, high, step = C._bounds(op)
                if C.ISSUE_GROUP in op.attributes:
                    raise ValueError("issue groups and carried completion need a separately composed target contract")
                initial = tuple(op.iter_args)
                if len(set(initial)) != len(initial) or any(v not in values for v in initial):
                    raise ValueError("loop completion inputs are aliased or do not dominate the loop")
                if set(pending) - set(initial):
                    raise ValueError("live completion crossing a loop must be an explicit iter_arg")
                states = tuple(pending.get(v) for v in initial)
                trips = (high - low + step - 1) // step
                seen, iteration = {}, 0
                if not trips:
                    # Even unreachable blocks must be well typed and closed;
                    # conservative footprint refusal is permitted, never a pass
                    # for an unchecked hidden command region.
                    body(op.body.block, states, local_names, yielding=True)
                while iteration < trips:
                    if states in seen:
                        period = iteration - seen[states]
                        skip = (trips - iteration) // period
                        if skip:
                            iteration += skip * period
                            continue
                    seen[states] = iteration
                    states = body(op.body.block, states, local_names, yielding=True)
                    iteration += 1
                summaries += 1
                for value in initial:
                    pending.pop(value, None)
                    values.pop(value)
                    names.pop(value)
                for value, state, name in zip(op.results, states, result_names, strict=True):
                    values[value], names[value] = state, name
                    if state is not None:
                        pending[value] = state
            elif type(op) is func.CallOp:
                metadata, operands = C._command(op, allow_persistent_state)
                declaration = declarations.get(op.callee.root_reference.data)
                if (declaration is None
                        or tuple(declaration.function_type.inputs) != tuple(v.type for v in op.arguments)
                        or tuple(declaration.function_type.outputs) != tuple(v.type for v in op.results)):
                    raise ValueError("command declaration and actual SSA call disagree")
                for name, event in zip(metadata.awaits, op.arguments[len(operands):], strict=True):
                    if event not in values or names[event] != name:
                        raise ValueError("event SSA edge differs from its dominating lexical binding")
                    pending.pop(event, None)
                    values[event] = None
                if any(C._overlap(a, b) for stages in pending.values() for a in stages for b in metadata.stages):
                    raise ValueError("private rows reused before their actual event dependency")
                if metadata.produces:
                    if metadata.produces in names.values():
                        raise ValueError("completion producer shadows a live lexical binding")
                    value = op.results[0]
                    values[value] = pending[value] = metadata.stages
                    names[value] = metadata.produces
            elif type(op) is scf.YieldOp:
                if not yielding or op is not block.last_op:
                    raise ValueError("completion yields require their lexical loop terminator")
                yielded = tuple(op.arguments)
                if (len(yielded) != len(incoming) or len(set(yielded)) != len(yielded)
                        or any(v not in values for v in yielded) or set(pending) - set(yielded)):
                    raise ValueError("every live event must be yielded once or completed before loop exit")
                return tuple(pending.get(v) for v in yielded)
            elif type(op) is func.ReturnOp:
                if yielding or op is not block.last_op or op.arguments:
                    raise ValueError("completion cannot escape the closed command entry")
            elif type(op) not in (arith.ConstantOp, arith.AddiOp, arith.MuliOp, arith.CmpiOp, arith.SelectOp):
                raise ValueError("unsupported operation in command control program")
            if type(op) in (arith.ConstantOp, arith.AddiOp, arith.MuliOp, arith.CmpiOp, arith.SelectOp):
                if (any(v.type not in (builtin.i64, builtin.i1) for v in (*op.operands, *op.results))
                        or type(op) is arith.CmpiOp and op.predicate.value.data != 0):
                    raise ValueError("unsupported command index type or predicate")
        if pending:
            raise ValueError("every launched event must complete before entry exit")
        if yielding:
            raise ValueError("completion loop requires its explicit yield terminator")
        return ()

    body(entry.body.block)
    return {"state_transfer_operations": work, "counted_loop_summaries": summaries,
            "method": "exact declared-stage transfer cycles; no dynamic trip expansion or device latency"}


def rename_ssa_events(module):
    """Alpha-rename metadata from actual SSA edges after a supported SCF clone.

    This changes no SSA operand, instruction, staging or completion effect.
    An edge not present in the lexical environment refuses; metadata never
    fabricates an edge. Callers must run the complete verifier afterwards.
    """
    from . import scf_commands as C

    counter = 0

    def fresh():
        nonlocal counter
        result = f"completion_{counter}"
        counter += 1
        return result

    def body(block, names):
        names = dict(names)
        for op in block.ops:
            if type(op) is scf.ForOp:
                local = tuple(fresh() for _ in op.iter_args)
                body(op.body.block, dict(zip(op.body.block.args[1:], local, strict=True)))
                results = tuple(fresh() for _ in op.results)
                if local:
                    op.attributes[CARRIED_EVENTS] = builtin.StringAttr(json.dumps(
                        {"inputs": local, "results": results}, sort_keys=True))
                for value in op.iter_args:
                    names.pop(value, None)
                names.update(zip(op.results, results, strict=True))
            elif type(op) is func.CallOp:
                data = json.loads(op.attributes[C.SCHEMA].data)
                try:
                    data["awaits"] = [names[v] for v in op.arguments[len(data["operands"]):]]
                except KeyError as error:
                    raise ValueError("cloned completion edge escaped its lexical scope") from error
                if op.results:
                    data["produces"] = fresh()
                    names[op.results[0]] = data["produces"]
                op.attributes[C.SCHEMA] = builtin.StringAttr(json.dumps(data, sort_keys=True, separators=(",", ":")))

    body(C._entry(module).body.block, {})
