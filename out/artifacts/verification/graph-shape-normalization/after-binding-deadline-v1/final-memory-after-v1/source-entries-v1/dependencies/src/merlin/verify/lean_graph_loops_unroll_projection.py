"""Complete actual inline/bulk/tail SSA decoding; trusted Python projection.

Forward bindings deliberately allow result aliases when an empty or forwarding
loop disappears. Clone-local bindings are separate from unchanged context SSA.
No witness, producer replay, or finite trace is used as proof authority.
"""

from __future__ import annotations

from .lean_graph_loops_segment_projection import _ids, _key, _metadata, _one_block, _types


class Decoder:
    def __init__(self, before, after, *, factor, index_bits, max_nodes):
        if type(factor) is not int or factor <= 1 or type(index_bits) is not int or index_bits not in (32, 64):
            raise ValueError("select a changed factor and signed32/64 context")
        self.factor, self.bits = factor, index_bits
        self.defs, self.target_defs, self.types = {}, {}, {}
        self.nodes = 0
        self.definitions(before, self.defs)
        self.definitions(after, self.target_defs)
        if type(max_nodes) is not int or not 0 < self.nodes <= max_nodes:
            raise ValueError("complete structural budget exceeded; no prefix accepted")
        self.env, self.plans = {}, []

    def definitions(self, op, definitions):
        _metadata(op)
        self.nodes += 1
        for identity, kind in op["results"]:
            if type(identity) is not int or identity < 0 or identity in definitions:
                raise ValueError("actual SSA definitions must be unique")
            definitions[identity] = op
            self.types[id(definitions), identity] = kind
        for region in op["regions"]:
            for block in region:
                if set(block) != {"arguments", "operations"}:
                    raise ValueError("unknown actual block fields")
                for identity, kind in block["arguments"]:
                    if type(identity) is not int or identity < 0 or identity in definitions:
                        raise ValueError("actual SSA arguments must be unique")
                    definitions[identity] = None
                    self.types[id(definitions), identity] = kind
                for child in block["operations"]:
                    self.definitions(child, definitions)

    @staticmethod
    def mapped(values, env):
        try:
            return [env[value] for value in values]
        except KeyError as error:
            raise ValueError("actual source operand/capture is not initialized in this scope") from error

    @staticmethod
    def bind(left, right, env):
        if _types(left) != _types(right):
            raise ValueError("complete SSA argument/result types or arities changed")
        for (source, _), (target, _) in zip(left, right, strict=True):
            if source in env:
                raise ValueError("source definition was rebound")
            env[source] = target

    def constant(self, value, *, target=False):
        op = (self.target_defs if target else self.defs).get(value)
        if (
            op is None
            or op["name"] != "arith.constant"
            or _types(op["results"]) != ["index"]
            or op["attributes"]
            or op["operands"]
            or op["regions"]
            or set(dict(op["properties"])) != {"value"}
            or type(op["index_constant"]) is not int
            or op["index_constant"] < 0
        ):
            raise ValueError("actual loop control requires a pure nonnegative index constant")
        return op["index_constant"]

    def operation(self, left, right, env, depth=0):
        if _metadata(left) != _metadata(right) or self.mapped(left["operands"], env) != right["operands"]:
            raise ValueError("unchanged operation/observer/capture/metadata changed")
        self.bind(left["results"], right["results"], env)
        if len(left["regions"]) != len(right["regions"]):
            raise ValueError("unchanged regions changed")
        a_regions, b_regions = [], []
        for a, b in zip(left["regions"], right["regions"], strict=True):
            if len(a) != len(b):
                raise ValueError("unchanged block structure changed")
            a_blocks, b_blocks = [], []
            for first, second in zip(a, b, strict=True):
                x, y = self.block(first, second, dict(env), depth + (left["name"] == "scf.for"))
                a_blocks.append(x)
                b_blocks.append(y)
            a_regions.append(a_blocks)
            b_regions.append(b_blocks)
        # Actual target operands were checked above, including noninjective aliases.
        first = dict(_metadata(left), operands=left["operands"], results=left["results"], regions=a_regions)
        second = dict(_metadata(right), operands=left["operands"], results=left["results"], regions=b_regions)
        return first, second

    def block(self, left, right, env, depth):
        self.bind(left["arguments"], right["arguments"], env)
        first, second, cursor = [], [], 0
        for op in left["operations"]:
            if op["name"] == "scf.for" and depth == 0:
                a, b, used = self.unroll(op, right["operations"][cursor:], env)
                cursor += used
            else:
                if cursor >= len(right["operations"]):
                    raise ValueError("unchanged operation is missing")
                a, b = self.operation(op, right["operations"][cursor], env, depth)
                cursor += 1
            first.append(a)
            second.append(b)
        if cursor != len(right["operations"]):
            raise ValueError("extra operations outside the complete changed boundary")
        return {"arguments": left["arguments"], "operations": first}, {
            "arguments": left["arguments"],
            "operations": second,
        }

    @staticmethod
    def yield_op(op):
        if op["name"] != "scf.yield" or any(op[k] for k in ("attributes", "properties", "regions", "results")):
            raise ValueError("a complete standard yield with no unknown semantics is required")
        return op["operands"]

    def fresh_constant(self, op, expected):
        if len(op["results"]) != 1 or self.constant(op["results"][0][0], target=True) != expected:
            raise ValueError("actual lane/bulk constant changed")
        return op["results"][0][0]

    def clone(self, body, operations, iv, state, parent):
        if len(operations) != len(body["operations"]) - 1:
            raise ValueError("clone operation count changed")
        env = dict(parent)
        source_args = body["arguments"]
        targets = [iv, *state]
        if len(source_args) != len(targets):
            raise ValueError("clone carry argument count changed")
        for (source, kind), target in zip(source_args, targets, strict=True):
            if source in env or self.types.get((id(self.target_defs), target)) != kind:
                raise ValueError("clone argument/capture typing changed")
            env[source] = target
        normalized = []
        for first, second in zip(body["operations"][:-1], operations, strict=True):
            if first["regions"] or second["regions"]:
                raise ValueError("nested body regions require an explicit semantics extension")
            a, b = self.operation(first, second, env, 1)
            if a != b:
                raise ValueError("entire cloned body changed")
            normalized.append(b)
        output = self.mapped(self.yield_op(body["operations"][-1]), env)
        # Body-key includes ALL args, operations and the complete original yield.
        canonical = {"arguments": source_args, "operations": [*normalized, body["operations"][-1]]}
        return output, _key(canonical)

    def loop_header(self, op, controls, initial, carry_types):
        if (
            op["name"] != "scf.for"
            or op["attributes"]
            or op["properties"]
            or op["operands"] != [*controls, *initial]
            or _types(op["results"]) != carry_types
        ):
            raise ValueError("actual bulk/tail bounds, captures or carry initialization changed")
        body = _one_block(op)
        if _types(body["arguments"]) != ["index", *carry_types] or not body["operations"]:
            raise ValueError("complete bulk/tail body arguments required")
        return body

    def unroll(self, source, emitted, parent):
        if source["attributes"] or source["properties"] or len(source["operands"]) < 3:
            raise ValueError("unknown source loop semantics")
        lower, upper, step = (self.constant(x) for x in source["operands"][:3])
        if lower > upper or step < 1:
            raise ValueError("unsupported source bounds")
        count = (upper - lower + step - 1) // step
        groups, remainder = divmod(count, self.factor)
        original = _one_block(source)
        if not original["operations"] or any(op["regions"] for op in original["operations"]):
            raise ValueError("region-free complete original body required")
        source_yield = self.yield_op(original["operations"][-1])
        carry_types = _types(source["results"])
        if (
            _types(original["arguments"]) != ["index", *carry_types]
            or len(source["operands"]) != 3 + len(carry_types)
            or len(source_yield) != len(carry_types)
        ):
            raise ValueError("source carry typing/binding changed")
        initial = self.mapped(source["operands"][3:], parent)
        state, clones, used = list(initial), [], 0
        body_length = len(original["operations"]) - 1
        body_key = _key(original)
        bulk_upper = lower + groups * self.factor * step
        if count <= self.factor:
            if len(emitted) < count * (body_length + 1):
                raise ValueError("inline body is incomplete")
            for lane in range(count):
                iv = self.fresh_constant(emitted[used], lower + lane * step)
                used += 1
                entry = list(state)
                state, clone_key = self.clone(original, emitted[used : used + body_length], iv, state, parent)
                used += body_length
                clones.append(
                    dict(index=lower + lane * step, source=body_key, target=clone_key, entry=entry, exit=state)
                )
            actual_exit = state
            bulk_body_key, tail_key = "", ""
            target_stride, target_bound = step * self.factor, bulk_upper
        else:
            if len(emitted) < 3:
                raise ValueError("complete bulk header missing")
            stride = self.fresh_constant(emitted[0], step * self.factor)
            bound = self.fresh_constant(emitted[1], bulk_upper)
            bulk = emitted[2]
            body = self.loop_header(
                bulk, [self.mapped(source["operands"][:1], parent)[0], bound, stride], initial, carry_types
            )
            target_stride, target_bound = self.constant(stride, target=True), self.constant(bound, target=True)
            bulk_body_key = _key(body)
            current = _ids(body["arguments"])[1:]
            bulk_iv, cursor = body["arguments"][0][0], 0
            bulk_initial = list(current)
            expected_length = self.factor * body_length + 2 * (self.factor - 1) + 1
            if len(body["operations"]) != expected_length:
                raise ValueError("dropped, duplicated or extra bulk lane operation")
            for lane in range(self.factor):
                iv = bulk_iv
                if lane:
                    offset, add = body["operations"][cursor : cursor + 2]
                    offset_value = self.fresh_constant(offset, lane * step)
                    if (
                        add["name"] != "arith.addi"
                        or add["attributes"]
                        or add["regions"]
                        or add["properties"] != [["overflowFlags", "#arith.overflow<none>"]]
                        or add["operands"] != [bulk_iv, offset_value]
                        or _types(add["results"]) != ["index"]
                    ):
                        raise ValueError("lane IV addition or offset binding changed")
                    iv = add["results"][0][0]
                    cursor += 2
                entry = list(current)
                current, clone_key = self.clone(
                    original, body["operations"][cursor : cursor + body_length], iv, current, parent
                )
                cursor += body_length
                clones.append(dict(index=lane * step, source=body_key, target=clone_key, entry=entry, exit=current))
            actual_yield = self.yield_op(body["operations"][-1])
            if actual_yield != current:
                raise ValueError("bulk yield lost or reordered carry values")
            state, used = _ids(bulk["results"]), 3
            tail_key = ""
            if remainder:
                if len(emitted) < 4:
                    raise ValueError("actual nonzero tail is missing")
                tail = emitted[3]
                tail_body = self.loop_header(
                    tail, [bound, *self.mapped(source["operands"][1:3], parent)], state, carry_types
                )
                if len(tail_body["operations"]) != body_length + 1:
                    raise ValueError("complete tail body changed")
                tail_state, tail_key = self.clone(
                    original,
                    tail_body["operations"][:-1],
                    tail_body["arguments"][0][0],
                    _ids(tail_body["arguments"])[1:],
                    parent,
                )
                if self.yield_op(tail_body["operations"][-1]) != tail_state:
                    raise ValueError("tail yield lost or reordered carry values")
                if _metadata(tail_body["operations"][-1]) != _metadata(original["operations"][-1]):
                    raise ValueError("cloned tail yield metadata changed")
                state, used = _ids(tail["results"]), 4
            actual_exit = list(state)
        if _types(source["results"]) != [self.types.get((id(self.target_defs), value)) for value in actual_exit]:
            raise ValueError("replacement result types changed")
        for (identity, _), value in zip(source["results"], actual_exit, strict=True):
            if identity in parent:
                raise ValueError("source loop result was already initialized")
            parent[identity] = value
        plan = dict(
            bounds=dict(
                bits=self.bits,
                lower=lower,
                upper=upper,
                step=step,
                factor=self.factor,
                count=count,
                groups=groups,
                remainder=remainder,
            ),
            sourceBody=body_key,
            clones=clones,
            sourceInitial=initial,
            targetInitial=initial,
            chainInitial=initial if count <= self.factor else bulk_initial,
            expectedExit=state if count <= self.factor else current,
            actualExit=actual_exit if count <= self.factor else actual_yield,
            sourceResults=carry_types,
            targetResults=[self.types[id(self.target_defs), x] for x in actual_exit],
            tailBody=tail_key,
            bulkBody=bulk_body_key,
            targetStride=target_stride,
            targetBound=target_bound,
            actualBindings=dict(
                source_results=source["results"],
                replacements=actual_exit,
                source_initial=source["operands"][3:],
                mapped_initial=initial,
            ),
        )
        self.plans.append(plan)
        hole = dict(boundary=len(self.plans) - 1, initial=source["operands"][3:], results=source["results"])
        # Complete actual target observer operands are checked by operation(), not
        # guessed from an inverse mapping that would be ambiguous for aliases.
        return hole, dict(hole), used


def project_trees(before, after, *, factor, index_bits, max_nodes):
    decoder = Decoder(before, after, factor=factor, index_bits=index_bits, max_nodes=max_nodes)
    source, target = decoder.operation(before, after, decoder.env)
    if len(decoder.plans) != 1:
        raise ValueError("exactly one complete actual source unrolling boundary required")
    plan = decoder.plans[0]
    plan.update(sourceContext=_key(source), targetContext=_key(target), structural_nodes=decoder.nodes)
    return plan
