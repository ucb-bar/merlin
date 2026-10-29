"""Independent complete SSA-DAG/multiplicity/anchor and current-order matching.

Total eligible primitives use the sealed closed whitelist, NOT get_effects
alone. All other operations, including regions and partial arithmetic, remain
anchored. These host facts restrict admission; Lean Code proofs carry semantics.
"""

from __future__ import annotations

import hashlib
import json

if __package__:
    from .lean_graph_state_scalar_capture import canonical, exact, metadata, selected_primitive
    from .lean_graph_state_schedule_errors import ScheduleDomainRefusal, ScheduleUnavailable, require
else:
    from lean_graph_state_scalar_capture import canonical, exact, metadata, selected_primitive
    from lean_graph_state_schedule_errors import ScheduleDomainRefusal, ScheduleUnavailable, require

# Audited static IR-domain messages in the reused projection only. Identity,
# source/runtime, producer or unexpected author ValueError is never in this set.
IR_DOMAIN = frozenset({
    "complete actual builtin.module pair required", "complete surrounding module metadata changed",
    "unsupported complete module block geometry", "complete nonempty function/declaration roster required",
    "unsupported module-level operation; no missing surrounding context", "duplicate actual function symbols",
    "missing/unsupported actual terminator", "complete function interface/attributes changed",
    "unsupported/dropped function CFG blocks", "actual operand lacks lexical availability/dominance",
    "complete graph exceeds explicit scalar intake budget", "unsupported actual CFG; no unreachable block can be dropped",
    "actual function argument slots differ from interface", "changed/unsupported terminator metadata or control",
    "unsupported loop metadata or block geometry", "supported actual SCF control needs a checked positive constant stride",
    "actual induction/carry argument slots differ", "actual loop result slots differ from carries",
    "unsupported region semantics; entire boundary refused",
    "internal-call/fuel composition bridge unavailable; external contracts only",
    "nested external symbol references require a separate call bridge", "only explicitly declared external calls are admitted",
    "actual call operand/result signature differs from its explicit declaration",
    "one common complete descriptor/type registry required", "inconsistent concrete control descriptor",
})


def domain_call(function, *args, **kwargs):
    """Narrow adapter for READONLY IR-only helpers; not an arbitrary error catch."""
    try:
        return function(*args, **kwargs)
    except ValueError as error:
        reason = str(error)
        if reason in IR_DOMAIN or reason.startswith((
            "unsupported complete parsed representation: ", "unsupported complete opaque attribute representation: ",
        )):
            raise ScheduleDomainRefusal(reason) from error
        raise ScheduleUnavailable("unexpected helper/identity/author failure: " + reason) from error


def describe(module):
    from xdsl.dialects import builtin, func, scf
    from xdsl.ir import Block, Operation

    require(type(module) is builtin.ModuleOp and len(module.body.blocks) == 1, "one complete builtin module block required")
    functions = list(module.body.block.ops)
    require(functions and all(type(op) is func.FuncOp for op in functions), "complete function/declaration roster required")
    require(len({op.sym_name.data for op in functions}) == len(functions), "duplicate function symbol")
    scopes, anchors, block_arguments, op_scopes = {}, {}, {}, {}
    primitive = {}
    count = 0

    def block_scope(block, path, captured):
        nonlocal count
        scopes[block] = path
        available = {*block.args, *captured}
        for index, arg in enumerate(block.args):
            block_arguments[arg] = {"scope": path, "argument": index, "type": domain_call(exact, arg.type)}
        anchor = 0
        for op in block.ops:
            count += 1
            require(count <= 2048, "complete scheduling input exceeds syntax budget2048")
            require(not op.successors, "unsupported CFG scheduling")
            require(all(value in available for value in op.operands), "actual operand violates lexical SSA dominance")
            op_scopes[op] = path
            selected = selected_primitive(op)
            primitive[op] = selected
            if selected is None:
                anchors[op] = [path, anchor]
                anchor += 1
            if op.regions:
                require(type(op) is scf.ForOp and len(op.regions) == 1 and len(op.body.blocks) == 1,
                        "unknown region operation refused; regions are not implicit total actions")
                require(selected is None, "region cannot be an eligible scalar primitive")
                block_scope(op.body.block, [*path, anchors[op][1], 0, 0], available)
            available.update(op.results)

    for i, function in enumerate(functions):
        require(len(function.body.blocks) <= 1, "unsupported multiblock function")
        if function.body.blocks:
            require(tuple(v.type for v in function.body.block.args) == tuple(function.function_type.inputs),
                    "actual entry arguments differ from signature")
            block_scope(function.body.block, [i], ())
    cache, active = {}, set()
    work = 0

    def value_key(value):
        if value in block_arguments:
            return {"block_argument": block_arguments[value]}
        require(isinstance(value.owner, Operation) and value.owner in op_scopes, "unbound actual result owner")
        return {"result": list(value.owner.results).index(value), "owner": node(value.owner)}

    def node(op):
        nonlocal work
        if op in cache:
            return cache[op]
        require(op not in active, "cyclic actual SSA/capture dependency")
        active.add(op)
        result = {"descriptor": domain_call(metadata, op), "operands": [value_key(v) for v in op.operands],
                  "eligible_total_primitive": primitive[op], "anchor": anchors.get(op),
                  "regions": [block_key(b) for region in op.regions for b in region.blocks]}
        active.remove(op)
        size = len(canonical(result))
        work += size
        require(size <= 1000000 and work <= 16000000, "explicit scheduling DAG serialization budget exceeded")
        cache[op] = result
        return result

    def block_key(block):
        rows = [node(op) for op in block.ops]
        return {"scope": scopes[block], "arguments": [domain_call(exact, arg.type) for arg in block.args],
                "anchors": [row for row in rows if row["anchor"] is not None],
                "pure_multiset": sorted([row for row in rows if row["anchor"] is None], key=canonical)}

    semantic = {"module": domain_call(metadata, module), "functions": [
        {"header": domain_call(metadata, function), "blocks": [block_key(block) for block in function.body.blocks]}
        for function in functions]}
    blocks = [block for op in module.walk() for region in op.regions for block in region.blocks]
    ordered = []
    for block in blocks:
        if block is module.body.block:
            ordered.append({"scope": ["module"], "operations": [domain_call(metadata, f) for f in functions]})
        else:
            ordered.append({"scope": scopes[block], "operations": [node(op) for op in block.ops]})
    return {"semantic": semantic, "blocks": ordered, "operations": count,
            "region_motion_admitted": False, "totality_from_empty_effects": False}


def dependencies(module, block_index):
    """Reconstruct planner-shaped STATIC facts; these are NOT execution laws."""
    from xdsl.ir import Operation
    from xdsl.traits import IsTerminator, get_effects
    from merlin.transforms.certificates import ir_text

    blocks = [block for op in module.walk() for region in op.regions for block in region.blocks]
    require(type(block_index) is int and 0 <= block_index < len(blocks), "current block index outside actual program")
    block = blocks[block_index]
    ops = list(block.ops)
    positions = {op: i for i, op in enumerate(ops)}
    edges, unknown = set(), set()
    previous = None
    for i, op in enumerate(ops):
        effectful = False
        for child in op.walk():
            for value in child.operands:
                owner = value.owner
                if not isinstance(owner, Operation):
                    continue
                while owner.parent is not block:
                    owner = owner.parent_op()
                    if owner is None:
                        break
                if owner is not None and owner is not op:
                    require(owner in positions, "current dependency owner outside block")
                    edges.add((positions[owner], i))
            terminal = child.has_trait(IsTerminator, value_if_unregistered=False)
            if terminal:
                if child is op:
                    edges.update((j, i) for j in range(i))
                continue
            effects = get_effects(child)
            if effects is None:
                unknown.add(i)
                effectful = True
            elif effects:
                effectful = True
        if effectful:
            if previous is not None:
                edges.add((previous, i))
            previous = i
    return {"block_index": block_index,
            "operation_sha256": [hashlib.sha256(ir_text(op).encode()).hexdigest() for op in ops],
            "edges": [list(e) for e in sorted(edges)], "unknown_effect_indices": sorted(unknown)}


def match_step(before, after, request):
    require(type(request) is dict and set(request) == {"block_index", "order", "witness"}, "closed current-order request required")
    index, order = request["block_index"], request["order"]
    require(type(index) is int and type(order) is list and all(type(i) is int for i in order), "exact integer order fields required")
    left, right = describe(before), describe(after)
    require(left["semantic"] == right["semantic"], "actual complete SSA-DAG/multiplicity/anchor context changed")
    require(len(left["blocks"]) == len(right["blocks"]) and 0 < index < len(left["blocks"]), "selected schedulable block is absent/module-level")
    source, target = left["blocks"][index], right["blocks"][index]
    require(sorted(order) == list(range(len(source["operations"]))), "current order is not a full permutation")
    require(source["scope"] == target["scope"] and [source["operations"][i] for i in order] == target["operations"],
            "actual output does not match current ordered SSA bindings")
    for i, (a, b) in enumerate(zip(left["blocks"], right["blocks"], strict=True)):
        if i != index:
            require(a == b, "unrequested block schedule changed")
    expected = {**dependencies(before, index), "order": order}
    witness = request["witness"]
    require(type(witness) is dict and set(witness) == set(expected), "closed recomputed dependency diagnostic required")
    # JSON equality alone is too permissive for bool-as-int numeric fields.
    require(canonical(witness) == canonical(expected), "fake dependency omission or stale current-order witness")
    positions = {node: i for i, node in enumerate(order)}
    require(all(positions[a] < positions[b] for a, b in expected["edges"]), "actual order violates recomputed static dependency")
    return {"before": left, "after": right, "request": request,
            "independent_static_facts": expected, "dependency_legality_is_execution_proof": False}
