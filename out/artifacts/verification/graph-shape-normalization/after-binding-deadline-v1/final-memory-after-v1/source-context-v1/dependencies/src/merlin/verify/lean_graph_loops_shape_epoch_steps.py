"""Complete typed statement-to-epoch instruction binding for the new adapter.

Every old instruction is paired with one actual ordinary Code atom or the final
return. This projection supplies explicit finite facts, not semantic authority;
whole-program simulation and qualification remain separate.
"""

from __future__ import annotations

import copy
import json

from .lean_graph_loops_segment_projection import _key, _one_block


def _opcode(table, key, row):
    if key in table and table[key] != row:
        raise ValueError("one actual operation key acquired inconsistent interpretations")
    table[key] = row


def _access(atom, epoch, buffer, indices):
    view_type = buffer["kind"]
    if view_type["tag"] != "view" or view_type["type"]["shape"] != epoch["shape"]:
        raise ValueError("typed private access geometry differs from actual epoch")
    return dict(
        object=epoch["object"],
        buffer=buffer,
        indices=indices,
        shape=epoch["shape"],
        element=view_type["type"]["element"],
    )


def bind_steps(adapter, normalized_export):
    memory = adapter["memory"]
    module = adapter["normalized_candidate"]
    mapping = {int(raw): source for raw, source in adapter["candidate_to_source"].items()}
    epochs = [
        dict(row, object=index, name=mapping[row["source_value"]])
        for index, row in enumerate(memory["reuse"]["epochs"])
    ]
    allocation_names = {epoch["name"] for epoch in epochs}
    candidates = [
        row
        for row in adapter["sequences"]
        if any(
            atom["tag"] == "action" and any(result["name"] in allocation_names for result in atom["results"])
            for atom in row["atoms"]
        )
    ]
    if len(candidates) != 1:
        raise ValueError("epoch adapter requires one complete function containing all private allocations")
    sequence = candidates[0]
    function = next(row for row in module["functions"] if row["name"] == sequence["function"])
    functions = list(
        zip(_one_block(normalized_export["tree"])["operations"], normalized_export["functions"], strict=True)
    )
    raw_function = next(op for op, prototype in functions if prototype["name"] == function["name"])
    raw_ops = _one_block(raw_function)["operations"]
    instructions = memory["reuse"]["instructions"]
    if len(instructions) != len(sequence["atoms"]) + 1 or len(raw_ops) != len(instructions):
        raise ValueError("complete atom/instruction/return boundary length differs")
    private_views = []
    for epoch in epochs:
        definitions = [
            result
            for atom in sequence["atoms"]
            if atom["tag"] == "action"
            for result in atom["results"]
            if result["name"] == epoch["name"]
        ]
        if len(definitions) != 1:
            raise ValueError("private descriptor has no unique actual definition")
        binding = definitions[0]
        private_views.append(
            dict(
                binding=binding,
                object=epoch["object"],
                shape=epoch["shape"],
                element=binding["kind"]["type"]["element"],
            )
        )
    specifications, opcodes, steps = {}, {}, []
    context, clock = copy.deepcopy(function["parameters"]), 0
    observers = {row["function"]: row for row in adapter["observers"]}
    targets = dict(module["calls"])
    for position, (atom, instruction, raw) in enumerate(
        zip(sequence["atoms"], instructions[:-1], raw_ops[:-1], strict=True)
    ):
        key = instruction.get("key")
        if key is not None and key != _key(raw):
            raise ValueError("epoch instruction does not identify its exact actual source operation")
        step = dict(position=position, atom=atom, instruction=instruction, context=copy.deepcopy(context), clock=clock)
        kind = instruction["kind"]
        results = [atom["result"]] if atom["tag"] == "constant" else atom["results"]
        spec = None
        if kind == "retained" and atom["tag"] == "constant":
            if raw["name"] != "arith.constant" or raw["index_constant"] != atom["value"]:
                raise ValueError("constant differs from its exact retained instruction")
            spec = dict(tag="constant", result=atom["result"], value=atom["value"])
        elif kind == "initialize":
            matches = [
                index
                for index, link in enumerate(adapter["initializer_links"])
                if link["config"]["object"] == instruction["object"] and link["config"]["key"] == key
            ]
            if atom["tag"] != "region" or len(matches) != 1:
                raise ValueError("region does not bind one complete initializer")
            step["initializer"] = matches[0]
            link = adapter["initializer_links"][matches[0]]
            _opcode(opcodes, link["canonical"]["storeKey"], dict(tag="store"))
            _opcode(opcodes, link["canonical"]["loopKey"], dict(tag="loop"))
        else:
            if atom["tag"] != "action":
                raise ValueError("epoch opcode disagrees with actual typed statement form")
            actual = json.loads(atom["key"])["name"]
            if kind in ("allocate", "release", "load", "store"):
                epoch = epochs[instruction["object"]]
                if kind == "allocate":
                    if actual != "memref.alloc" or atom["operands"] or len(results) != 1:
                        raise ValueError("unsupported actual private allocation")
                    if results[0]["name"] != epoch["name"] or epoch["object"] != clock:
                        raise ValueError("logical allocator count does not match source allocation order")
                    clock += 1
                    _opcode(
                        opcodes,
                        atom["key"],
                        dict(tag="allocate", shape=epoch["shape"], element=results[0]["kind"]["type"]["element"]),
                    )
                else:
                    expected_name = {"release": "memref.dealloc", "load": "memref.load", "store": "memref.store"}[kind]
                    if actual != expected_name:
                        raise ValueError("private memory opcode changed")
                    buffer_position = 1 if kind == "store" else 0
                    if (
                        len(atom["operands"]) <= buffer_position
                        or atom["operands"][buffer_position]["name"] != epoch["name"]
                    ):
                        raise ValueError("private access targets another actual descriptor")
                    _opcode(opcodes, atom["key"], dict(tag=kind))
                    if kind == "load":
                        if len(results) != 1:
                            raise ValueError("private load result arity changed")
                        spec = dict(
                            tag="load",
                            access=_access(atom, epoch, atom["operands"][0], atom["operands"][1:]),
                            result=results[0],
                        )
                    elif kind == "store":
                        if results:
                            raise ValueError("private store has an unexpected result")
                        spec = dict(
                            tag="store",
                            access=_access(atom, epoch, atom["operands"][1], atom["operands"][2:]),
                            value=atom["operands"][0],
                        )
                    elif results or len(atom["operands"]) != 1:
                        raise ValueError("release operand/result arity changed")
            elif kind == "retained" and actual == "arith.addi":
                if len(atom["operands"]) != 2 or len(results) != 1 or results[0]["kind"]["tag"] != "scalar":
                    raise ValueError("unsupported scalar binary signature")
                element = results[0]["kind"]["type"]
                spec = dict(
                    tag="binary",
                    operation=atom["key"],
                    left=atom["operands"][0],
                    right=atom["operands"][1],
                    result=results[0],
                    element=element,
                )
                _opcode(opcodes, atom["key"], dict(tag="binary", element=element))
            elif kind == "retained" and actual == "func.call":
                target = targets.get(atom["key"])
                if target not in observers or len(atom["operands"]) != 2 or len(results) != 1:
                    raise ValueError("retained call lacks a derived complete caller-observer body")
                observer = observers[target]
                spec = dict(
                    tag="observer",
                    operation=observer["binaryKey"],
                    input=atom["operands"][0],
                    buffer=atom["operands"][1],
                    result=results[0],
                    element=observer["element"],
                )
                step["observer"] = target
                for opcode_name, tag in (("loadKey", "load"), ("storeKey", "store"), ("binaryKey", "binary")):
                    row = dict(tag=tag)
                    if tag == "binary":
                        row["element"] = observer["element"]
                    _opcode(opcodes, observer[opcode_name], row)
            else:
                raise ValueError("unsupported complete retained statement in epoch adapter")
        if spec is not None:
            if key in specifications and specifications[key] != spec:
                raise ValueError("one raw instruction key acquired different SSA operands")
            specifications[key] = spec
        for result in results:
            if any(bound["name"] == result["name"] for bound in context):
                raise ValueError("top-level SSA result is not fresh")
            context.insert(0, result)
        step.update(afterContext=copy.deepcopy(context), afterClock=clock)
        steps.append(step)
    terminal = instructions[-1]
    if terminal["kind"] != "retained" or terminal["key"] != _key(raw_ops[-1]) or raw_ops[-1]["name"] != "func.return":
        raise ValueError("complete epoch return boundary changed")
    specifications[terminal["key"]] = dict(tag="returned", values=sequence["returned"])
    return dict(
        adapter,
        step_binding=dict(
            function=function["name"],
            steps=steps,
            privateViews=private_views,
            specifications=sorted(specifications.items()),
            opcodes=sorted(opcodes.items()),
            finalContext=context,
            finalClock=clock,
            returned=sequence["returned"],
            terminal=terminal,
        ),
        full_epoch_public_adapter_proved=False,
    )
