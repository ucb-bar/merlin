"""Independent ordinary final-program projection, including dynamic SCF bounds.

This is an additive decoder for the already-normalized final program. It does
not consume a source tree, reuse plan, segmentation witness or whole-tree match.
Its output is a candidate ordinary typed Code term, not a verified memory join.
Actual loop operand SSA remains explicit even when its value is dynamic.
"""

from __future__ import annotations

from .lean_graph_loops_segment_projection import _key, _one_block, _types
from .lean_graph_loops_shape_after_projection import SingleDecoder
from .lean_graph_loops_shape_types import binding, index_constant, operation_key, standard_yield


class FinalDecoder(SingleDecoder):
    def __init__(self, exported, *, index_bits, max_nodes):
        super().__init__(exported, index_bits=index_bits, max_nodes=max_nodes)
        self.index_opcodes = {}
        self.dynamic_regions = 0

    def ordinary_block(self, block, context, *, terminal):
        if not block["operations"]:
            raise ValueError("complete final block requires a terminator")
        self.define(block["arguments"], context)
        commands = []
        for position, op in enumerate(block["operations"]):
            if position == len(block["operations"]) - 1:
                standard_yield(op, terminal)
                code = dict(tag="returned", values=self.read(op["operands"], context))
                break
            name, operands = op["name"], self.read(op["operands"], context)
            if name in (
                "memref.cast",
                "memref.dim",
                "memref.subview",
                "memref.reinterpret_cast",
                "memref.view",
                "memref.reshape",
            ):
                raise ValueError("final memory subset requires normalized static descriptors")
            if name == "arith.constant" and op["index_constant"] is not None:
                value = index_constant(op, self.types)
                result = self.define(op["results"], context, known=value)[0]
                commands.append(dict(tag="constant", result=result, value=value))
            elif op["regions"]:
                if name != "scf.for" or op["attributes"] or op["properties"] or len(operands) < 3:
                    raise ValueError("final memory region outside the single-body SCF subset")
                if any(row["kind"] != dict(tag="index") for row in operands[:3]):
                    raise ValueError("SCF controls must be index bindings")
                if operands[2]["known"] is None or operands[2]["known"] <= 0:
                    raise ValueError("final SCF requires a known positive stride")
                self.dynamic_regions += any(row["known"] is None for row in operands[:2])
                body = _one_block(op)
                if _types(body["arguments"]) != ["index", *_types(op["results"])]:
                    raise ValueError("final SCF induction/carry signature differs")
                parameters = [binding(identity, self.types) for identity, _ in body["arguments"]]
                nested = self.ordinary_block(body, dict(context), terminal="scf.yield")
                results = self.define(op["results"], context)
                commands.append(
                    dict(
                        tag="region",
                        key=operation_key(op),
                        operands=operands,
                        parameters=parameters,
                        results=results,
                        body=nested,
                    )
                )
            else:
                results = self.define(op["results"], context)
                key = operation_key(op)
                if name in ("arith.addi", "arith.minui") and any(row["kind"] == dict(tag="index") for row in results):
                    expected_properties = [["overflowFlags", "#arith.overflow<none>"]] if name == "arith.addi" else []
                    if op["attributes"] or op["properties"] != expected_properties:
                        raise ValueError("generated index control metadata outside the explicit interpretation")
                    if (
                        len(operands) != 2
                        or len(results) != 1
                        or any(row["kind"] != dict(tag="index") for row in [*operands, *results])
                    ):
                        raise ValueError("generated index control requires two index operands and one index result")
                    opcode = "add" if name == "arith.addi" else "minimum"
                    if key in self.index_opcodes and self.index_opcodes[key] != opcode:
                        raise ValueError("ambiguous generated control operation key")
                    self.index_opcodes[key] = opcode
                commands.append(dict(tag="action", key=key, operands=operands, results=results))
                if name == "func.call":
                    target = self.call_targets.get(_key(op))
                    if target is None or target["nested"]:
                        raise ValueError("final call has no admitted flat target")
                    self.calls.append([key, target["target"]])
        for command in reversed(commands):
            code = dict(command, rest=code)
        return code


def decode_final(exported, *, index_bits, max_nodes):
    decoder = FinalDecoder(exported, index_bits=index_bits, max_nodes=max_nodes)
    module = decoder.ordinary_module()
    return dict(
        module=module,
        frame=decoder.frame,
        index_opcodes=sorted(decoder.index_opcodes.items()),
        dynamic_regions=decoder.dynamic_regions,
        independent_actual_final=True,
        source_blueprint_consumed=False,
        whole_tree_match_used=False,
        operational_bridge_proved=False,
        lean_checked=False,
    )
