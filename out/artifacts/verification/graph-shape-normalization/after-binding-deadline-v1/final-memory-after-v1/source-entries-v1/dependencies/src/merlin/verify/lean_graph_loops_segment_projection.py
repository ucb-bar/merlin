"""Complete structural/SSA projection of a supported actual SCF segmentation.

Consumes full trees freshly exported from actual before/after modules. The source
loop, every clone/capture/yield, and every untouched operation are inspected.
Canonical body/context syntax has an arbitrary explicitly admitted interpretation
in Lean. This decoder is trusted Python, not a formally proved MLIR parser.
"""

from __future__ import annotations

import json

OP_FIELDS = {"name", "operands", "results", "attributes", "properties", "location", "regions", "index_constant"}


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _metadata(op):
    if set(op) != OP_FIELDS:
        raise ValueError("unknown operation fields need an explicit projection extension")
    return {name: op[name] for name in ("name", "attributes", "properties", "location", "index_constant")}


def _one_block(op):
    if len(op["regions"]) != 1 or len(op["regions"][0]) != 1:
        raise ValueError("supported loop requires exactly one actual body block")
    return op["regions"][0][0]


def _types(values):
    return [value[1] for value in values]


def _ids(values):
    return [value[0] for value in values]


class _Projection:
    def __init__(self, before, after, *, tile, index_bits, max_nodes):
        if type(tile) is not int or tile <= 1 or type(index_bits) is not int or index_bits not in (32, 64):
            raise ValueError("select an actual changed tile and signed32/64 indices")
        self.tile, self.bits = tile, index_bits
        self.before_defs, self.after_defs = {}, {}
        self.count = 0
        self._definitions(before, self.before_defs)
        self._definitions(after, self.after_defs)
        if type(max_nodes) is not int or max_nodes <= 0 or self.count > max_nodes:
            raise ValueError("complete structural node budget exceeded; no prefix accepted")
        self.mapping = {}
        self.inverse_mapping = {}
        self.plans = []

    def _definitions(self, op, definitions):
        _metadata(op)
        self.count += 1
        for identity, _type in op["results"]:
            if identity in definitions:
                raise ValueError("duplicate actual SSA definition")
            definitions[identity] = op
        for region in op["regions"]:
            for block in region:
                if set(block) != {"arguments", "operations"}:
                    raise ValueError("unknown actual block fields")
                for identity, _type in block["arguments"]:
                    if identity in definitions:
                        raise ValueError("duplicate actual SSA argument")
                    definitions[identity] = None
                for child in block["operations"]:
                    self._definitions(child, definitions)

    def _bind(self, left, right):
        if _types(left) != _types(right):
            raise ValueError("actual SSA argument/result types or arity changed")
        for (source, _), (target, _) in zip(left, right, strict=True):
            if source in self.mapping or target in self.inverse_mapping:
                raise ValueError("SSA correspondence must preserve unique definitions")
            self.mapping[source] = target
            self.inverse_mapping[target] = source

    def _mapped(self, values):
        try:
            return [self.mapping[value] for value in values]
        except KeyError as error:
            raise ValueError("actual operation uses an unbound source capture") from error

    def _inverse(self, values):
        try:
            return [self.inverse_mapping[value] for value in values]
        except KeyError as error:
            raise ValueError("target capture has no corresponding actual source value") from error

    def _constant(self, value, *, after=False):
        op = (self.after_defs if after else self.before_defs).get(value)
        if (
            op is None
            or op["name"] != "arith.constant"
            or _types(op["results"]) != ["index"]
            or op["attributes"]
            or op["operands"]
            or op["regions"]
            or set(dict(op["properties"])) != {"value"}
        ):
            raise ValueError("loop control must use actual static index constants")
        result = op["index_constant"]
        if type(result) is not int or result < 0:
            raise ValueError("negative or unknown static loop bounds are outside this family")
        return result

    def _eligible(self, op, depth):
        if op["name"] != "scf.for" or depth != 0:
            return False
        try:
            lower, upper, step = [self._constant(value) for value in op["operands"][:3]]
        except ValueError:
            return False
        return lower <= upper and step > 0

    def operation(self, left, right, depth=0):
        if _metadata(left) != _metadata(right) or self._mapped(left["operands"]) != right["operands"]:
            raise ValueError("actual untouched operation, capture, attribute or property changed")
        self._bind(left["results"], right["results"])
        if len(left["regions"]) != len(right["regions"]):
            raise ValueError("actual unchanged region boundary changed")
        left_regions, right_regions = [], []
        for first, second in zip(left["regions"], right["regions"], strict=True):
            if len(first) != len(second):
                raise ValueError("actual block/control structure changed")
            first_blocks, second_blocks = [], []
            for a, b in zip(first, second, strict=True):
                x, y = self.block(a, b, depth + (left["name"] == "scf.for"))
                first_blocks.append(x)
                second_blocks.append(y)
            left_regions.append(first_blocks)
            right_regions.append(second_blocks)
        first = dict(_metadata(left), operands=left["operands"], results=left["results"], regions=left_regions)
        second = dict(
            _metadata(right),
            operands=self._inverse(right["operands"]),
            results=list(zip(self._inverse(_ids(right["results"])), _types(right["results"]), strict=True)),
            regions=right_regions,
        )
        return first, second

    def block(self, left, right, depth):
        self._bind(left["arguments"], right["arguments"])
        first, second, cursor = [], [], 0
        for op in left["operations"]:
            if self._eligible(op, depth):
                if cursor + 1 >= len(right["operations"]):
                    raise ValueError("complete segmented boundary is missing")
                a, b = self.segment(op, right["operations"][cursor], right["operations"][cursor + 1])
                cursor += 2
            else:
                if cursor >= len(right["operations"]):
                    raise ValueError("an actual unchanged operation is missing")
                a, b = self.operation(op, right["operations"][cursor], depth)
                cursor += 1
            first.append(a)
            second.append(b)
        if cursor != len(right["operations"]):
            raise ValueError("extra actual operations are outside the complete changed boundary")
        return {"arguments": left["arguments"], "operations": first}, {
            "arguments": list(zip(self._inverse(_ids(right["arguments"])), _types(right["arguments"]), strict=True)),
            "operations": second,
        }

    def segment(self, source, stride, outer):
        if source["attributes"] or source["properties"]:
            raise ValueError("unmodeled source loop attributes/properties refuse")
        if (
            stride["name"] != "arith.constant"
            or _types(stride["results"]) != ["index"]
            or stride["attributes"]
            or outer["name"] != "scf.for"
            or outer["attributes"]
            or outer["properties"]
            or outer["location"] != source["location"]
        ):
            raise ValueError("actual outer loop/stride is not the supported pure index control")
        if len(source["operands"]) < 3 or len(outer["operands"]) < 3:
            raise ValueError("actual loop control operands are incomplete")
        lower, upper, step = [self._constant(value) for value in source["operands"][:3]]
        if (
            outer["operands"][:2] != self._mapped(source["operands"][:2])
            or outer["operands"][2] != stride["results"][0][0]
        ):
            raise ValueError("actual outer bound/stride SSA binding changed")
        target_lower, target_upper, target_stride = [
            self._constant(value, after=True) for value in outer["operands"][:3]
        ]
        initial = source["operands"][3:]
        outer_initial = self._inverse(outer["operands"][3:])
        self._bind(source["results"], outer["results"])
        source_body, outer_body = _one_block(source), _one_block(outer)
        if len(outer_body["operations"]) != 4 or not outer_body["arguments"]:
            raise ValueError("all add/min/inner/yield operations and actual outer IV are required")
        add, minimum, inner, final_yield = outer_body["operations"]
        if (
            add["name"] != "arith.addi"
            or add["attributes"]
            or add["regions"]
            or add["properties"] != [["overflowFlags", "#arith.overflow<none>"]]
            or _types(add["results"]) != ["index"]
            or minimum["name"] != "arith.minui"
            or minimum["attributes"]
            or minimum["properties"]
            or minimum["regions"]
            or _types(minimum["results"]) != ["index"]
            or inner["name"] != "scf.for"
            or inner["attributes"]
            or inner["properties"]
            or final_yield["name"] != "scf.yield"
            or final_yield["attributes"]
            or final_yield["properties"]
            or final_yield["regions"]
            or final_yield["results"]
        ):
            raise ValueError("unknown or incomplete generated control semantics")
        inner_body = _one_block(inner)
        if any(op["regions"] for op in source_body["operations"]):
            raise ValueError("nested source body regions require a separate admitted semantics")
        if not source_body["operations"] or source_body["operations"][-1]["name"] != "scf.yield":
            raise ValueError("complete source body yield is required")
        left_body, right_body = self.block(source_body, inner_body, 1)
        carry_types = _types(source_body["arguments"])[1:]
        plan = {
            "bounds": {"bits": self.bits, "lower": lower, "upper": upper, "step": step, "tile": self.tile},
            "targetLower": target_lower,
            "targetUpper": target_upper,
            "targetStride": target_stride,
            "targetInnerStep": self._constant(inner["operands"][2], after=True),
            "sourceBody": _key(left_body),
            "targetBody": _key(right_body),
            "carryTypes": carry_types,
            "sourceInitial": initial,
            "outerInitial": outer_initial,
            "sourceResults": _types(source["results"]),
            "outerResults": _types(outer["results"]),
            "innerResults": _types(inner["results"]),
            "outerArgumentTypes": _types(outer_body["arguments"]),
            "outerIV": outer_body["arguments"][0][0],
            "outerCarryArguments": _ids(outer_body["arguments"])[1:],
            "strideValue": stride["results"][0][0],
            "upperValue": outer["operands"][1],
            "originalStepValue": self._mapped([source["operands"][2]])[0],
            "addOperands": add["operands"],
            "addResult": add["results"][0][0],
            "minOperands": minimum["operands"],
            "minResult": minimum["results"][0][0],
            "innerOperands": inner["operands"],
            "innerResultValues": _ids(inner["results"]),
            "outerYield": final_yield["operands"],
            "sourceYieldArity": len(source_body["operations"][-1]["operands"]),
        }
        self.plans.append(plan)
        hole = {"boundary": len(self.plans) - 1, "initial": initial, "results": source["results"]}
        after_hole = {
            "boundary": len(self.plans) - 1,
            "initial": outer_initial,
            "results": list(zip(self._inverse(_ids(outer["results"])), _types(outer["results"]), strict=True)),
        }
        return hole, after_hole


def project_trees(before: dict, after: dict, *, tile: int, index_bits: int, max_nodes: int) -> dict:
    decoder = _Projection(before, after, tile=tile, index_bits=index_bits, max_nodes=max_nodes)
    first, second = decoder.operation(before, after)
    if len(decoder.plans) != 1:
        raise ValueError("this complete-boundary mode requires exactly one actual segmented source loop")
    result = decoder.plans[0]
    result.update(sourceContext=_key(first), targetContext=_key(second), structural_nodes=decoder.count)
    return result
