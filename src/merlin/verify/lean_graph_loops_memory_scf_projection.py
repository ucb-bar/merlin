"""Complete actual reuse/multiple nested-segmentation binding, trusted Python.

Every original, exact middle and final operation is matched. Scoped forward SSA
maps avoid inverse alias ambiguity. Only rank1/rank2 private initializers are
admitted; the operational Lean model interprets each pure op and actual store.
"""

from __future__ import annotations

from .lean_graph_loops_reuse_projection import Decoder
from .lean_graph_loops_segment_projection import _ids, _key, _metadata, _one_block, _types


class ReuseBinding(Decoder):
    """Keep actual matched middle initializers as well as the old complete model."""

    def __init__(self, *args, **kwargs):
        self.middle_initializers = {}
        super().__init__(*args, **kwargs)

    def operation(self, left, right, env, path=()):
        result = super().operation(left, right, env, path)
        if path in self.initializers:
            self.middle_initializers[self.initializers[path]] = right
        return result

    def complete(self):
        first, second = self.operation(self.before["tree"], self.after["tree"], self.env)
        if self.live_targets or any(epoch["slot"] is None for epoch in self.epochs):
            raise ValueError("missing final physical free or unbound logical epoch")
        return dict(
            bits=self.bits,
            sourceContext=_key(first),
            targetContext=_key(second),
            objects=[
                {key: epoch[key] for key in ("slot", "shape", "elementBits", "alignment")} for epoch in self.epochs
            ],
            instructions=self.instructions,
            epochs=self.epochs,
            structural_nodes=self.nodes,
            actual_slot_values=self.slots,
            allocator_mode="total_private_unobservable",
            initializer_semantics="partial_complete_rectangular_pure",
            raw_SCF_bridge_proved=False,
        )


def initializer_program(decoder, epoch):
    """Decode explicit operations/SSA, not an admitted complete array summary."""
    if len(epoch["shape"]) not in (1, 2):
        raise ValueError("operation-level initializer bridge supports rank1/rank2 only")
    current = decoder.source_ops[tuple(epoch["initializer"]["path"])]
    ivs, prefixes = [], []
    for _ in epoch["shape"]:
        body = _one_block(current)
        ivs.append(body["arguments"][0][0])
        prefix = []
        for op in body["operations"][:-2]:
            if any(kind not in ("index", "i8", "i16", "i32", "i64") for _, kind in op["results"]):
                raise ValueError("pure initializer result needs an explicit scalar/index interpretation")
            prefix.append(dict(key=_key(op), operands=op["operands"], results=_ids(op["results"])))
        prefixes.append(prefix)
        current = body["operations"][-2]
    return dict(
        shape=epoch["shape"],
        ivs=ivs,
        outerPrefix=prefixes[0] if len(ivs) == 2 else [],
        leaf=dict(ops=prefixes[-1], value=current["operands"][0], indices=current["operands"][2:]),
    )


class SegmentBinding:
    """Match complete middle/final trees, replacing ALL chosen loop boundaries."""

    def __init__(self, before, after, expected, *, index_bits, max_nodes):
        self.expected, self.bits = expected, index_bits
        self.source_defs, self.target_defs, self.plans, self.nodes = {}, {}, [], 0
        self.collect(before, self.source_defs)
        self.collect(after, self.target_defs)
        if type(max_nodes) is not int or not 0 < self.nodes <= max_nodes:
            raise ValueError("complete segmentation node budget exceeded")

    def collect(self, op, definitions):
        _metadata(op)
        self.nodes += 1
        for identity, kind in op["results"]:
            if identity in definitions:
                raise ValueError("duplicate actual SSA definition")
            definitions[identity] = (op, kind)
        for region in op["regions"]:
            for block in region:
                if set(block) != {"arguments", "operations"}:
                    raise ValueError("unknown actual block fields")
                for identity, kind in block["arguments"]:
                    if identity in definitions:
                        raise ValueError("duplicate actual block argument")
                    definitions[identity] = (None, kind)
                for child in block["operations"]:
                    self.collect(child, definitions)

    @staticmethod
    def mapped(values, env):
        try:
            return [env[value] for value in values]
        except KeyError as error:
            raise ValueError("unbound actual SSA/capture") from error

    @staticmethod
    def bind(first, second, env):
        if _types(first) != _types(second):
            raise ValueError("SSA result/argument arity/type changed")
        for source, target in zip(_ids(first), _ids(second), strict=True):
            if source in env:
                raise ValueError("SSA source definition rebound")
            env[source] = target

    def constant(self, identity, *, target=False):
        op, kind = (self.target_defs if target else self.source_defs).get(identity, (None, None))
        if (
            op is None
            or kind != "index"
            or op["name"] != "arith.constant"
            or op["attributes"]
            or op["operands"]
            or op["regions"]
            or set(dict(op["properties"])) != {"value"}
            or len(op["results"]) != 1
            or type(op["index_constant"]) is not int
            or op["index_constant"] < 0
        ):
            raise ValueError("static nonnegative exact index constant required")
        return op["index_constant"]

    def operation(self, first, second, env):
        if _metadata(first) != _metadata(second) or self.mapped(first["operands"], env) != second["operands"]:
            raise ValueError("retained operation/observer/capture/metadata changed")
        self.bind(first["results"], second["results"], env)
        if len(first["regions"]) != len(second["regions"]):
            raise ValueError("retained region count changed")
        a_regions, b_regions = [], []
        for a, b in zip(first["regions"], second["regions"], strict=True):
            if len(a) != len(b):
                raise ValueError("retained block count changed")
            pairs = [self.block(x, y, dict(env)) for x, y in zip(a, b, strict=True)]
            a_regions.append([x for x, _ in pairs])
            b_regions.append([y for _, y in pairs])
        common = dict(_metadata(first), operands=first["operands"], results=first["results"])
        return dict(common, regions=a_regions), dict(common, regions=b_regions)

    def block(self, first, second, env):
        self.bind(first["arguments"], second["arguments"], env)
        before, after, cursor = [], [], 0
        for op in first["operations"]:
            if _key(op) in self.expected:
                if cursor + 1 >= len(second["operations"]):
                    raise ValueError("missing complete generated initializer boundary")
                a, b = self.segment(op, *second["operations"][cursor : cursor + 2], env)
                cursor += 2
            else:
                if cursor >= len(second["operations"]):
                    raise ValueError("missing unchanged operation")
                a, b = self.operation(op, second["operations"][cursor], env)
                cursor += 1
            before.append(a)
            after.append(b)
        if cursor != len(second["operations"]):
            raise ValueError("extra operation outside complete boundary")
        return dict(arguments=first["arguments"], operations=before), dict(
            arguments=first["arguments"], operations=after
        )

    def segment(self, source, stride, outer, env):
        epoch = self.expected[_key(source)]
        if (
            source["name"] != "scf.for"
            or source["attributes"]
            or source["properties"]
            or source["results"]
            or len(source["operands"]) != 3
            or _metadata(outer) != _metadata(source)
            or outer["results"]
            or len(outer["operands"]) != 3
        ):
            raise ValueError("supported exact no-carry initializer control required")
        lower, upper, step = [self.constant(value) for value in source["operands"]]
        if len(stride["results"]) != 1:
            raise ValueError("one generated stride SSA value required")
        stride_value = stride["results"][0][0]
        actual_stride = self.constant(stride_value, target=True)
        if step != 1 or lower != 0 or actual_stride <= 1:
            raise ValueError("rectangular unit step and actual changed segmentation tile required")
        if outer["operands"] != [*self.mapped(source["operands"][:2], env), stride_value]:
            raise ValueError("generated outer bounds/stride binding changed")
        source_body, outer_body = _one_block(source), _one_block(outer)
        if (
            _types(source_body["arguments"]) != ["index"]
            or _types(outer_body["arguments"]) != ["index"]
            or len(outer_body["operations"]) != 4
        ):
            raise ValueError("exact outer IV, add/min/inner/yield required")
        add, minimum, inner, final_yield = outer_body["operations"]
        for op, name, properties in (
            (add, "arith.addi", [["overflowFlags", "#arith.overflow<none>"]]),
            (minimum, "arith.minui", []),
        ):
            if (
                op["name"] != name
                or op["attributes"]
                or op["properties"] != properties
                or op["regions"]
                or _types(op["results"]) != ["index"]
            ):
                raise ValueError("generated bound arithmetic semantics changed")
        if _metadata(inner) != _metadata(source) or inner["results"] or len(inner["operands"]) != 3:
            raise ValueError("generated inner loop semantics changed")
        Decoder.standard_yield(final_yield)
        inner_body = _one_block(inner)
        Decoder.standard_yield(source_body["operations"][-1])
        left, right = self.block(source_body, inner_body, dict(env))
        plan = dict(
            bounds=dict(bits=self.bits, lower=lower, upper=upper, step=step, tile=actual_stride),
            targetLower=self.constant(outer["operands"][0], target=True),
            targetUpper=self.constant(outer["operands"][1], target=True),
            targetStride=actual_stride,
            targetInnerStep=self.constant(inner["operands"][2], target=True),
            sourceBody=_key(left),
            targetBody=_key(right),
            carryTypes=[],
            sourceInitial=[],
            outerInitial=[],
            sourceResults=[],
            outerResults=[],
            innerResults=[],
            outerArgumentTypes=["index"],
            outerIV=outer_body["arguments"][0][0],
            outerCarryArguments=[],
            strideValue=stride_value,
            upperValue=outer["operands"][1],
            originalStepValue=self.mapped([source["operands"][2]], env)[0],
            addOperands=add["operands"],
            addResult=add["results"][0][0],
            minOperands=minimum["operands"],
            minResult=minimum["results"][0][0],
            innerOperands=inner["operands"],
            innerResultValues=[],
            outerYield=final_yield["operands"],
            sourceYieldArity=0,
        )
        self.plans.append(dict(epoch=epoch, middleInitializer=_key(source), projection=plan))
        hole = dict(initializer_boundary=epoch)
        return hole, hole


def project_exports(before, middle, after, *, index_bits, max_nodes):
    reuse = ReuseBinding(before, middle, index_bits=index_bits, max_nodes=max_nodes)
    original_model = reuse.complete()
    expected = {_key(op): epoch for epoch, op in reuse.middle_initializers.items()}
    if len(expected) != len(reuse.epochs):
        raise ValueError("missing or ambiguous actual middle initializer")
    segment = SegmentBinding(middle["tree"], after["tree"], expected, index_bits=index_bits, max_nodes=max_nodes)
    first, last = segment.operation(middle["tree"], after["tree"], {})
    if sorted(row["epoch"] for row in segment.plans) != list(range(len(reuse.epochs))):
        raise ValueError("every initializer must be matched exactly once, no overlapping selected boundary")
    configs = []
    for row in sorted(segment.plans, key=lambda item: item["epoch"]):
        epoch = row["epoch"]
        plan = row["projection"]
        plan.update(sourceContext=_key(first), targetContext=_key(last))
        configs.append(
            dict(
                object=epoch,
                key=reuse.epochs[epoch]["initializer"]["body"],
                program=initializer_program(reuse, reuse.epochs[epoch]),
                tile=plan["bounds"]["tile"],
                middleInitializer=row["middleInitializer"],
                segment=plan,
            )
        )
    return dict(
        reuse=original_model,
        configurations=configs,
        middleBinding=_key(middle["tree"]),
        structural_nodes=reuse.nodes + segment.nodes,
        python_projection_trusted=True,
        raw_parser_formally_verified=False,
    )
