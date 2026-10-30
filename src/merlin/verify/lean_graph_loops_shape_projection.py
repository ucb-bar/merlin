"""Independent complete actual shape-normalization module/SSA matcher.

No producer replay or saved witness is an authority. Every source/candidate
operation, nested scope, function/callee, observer and result is matched. The
blueprint preserves explicit source bindings and forward candidate aliases.
"""

from __future__ import annotations

from .lean_graph_input_admission import ModelRefusal
from .lean_graph_loops_segment_projection import _key, _metadata, _one_block, _types
from .lean_graph_loops_shape_types import binding, index_constant, kind, operation_key, standard_yield


class Decoder:
    def __init__(self, before, after, *, index_bits, max_nodes):
        if type(index_bits) is not int or index_bits not in (32, 64):
            raise ValueError("explicit signed32/64 index domain required")
        self.before, self.after, self.bits = before, after, index_bits
        self.types, self.target_types = before["types"], after["types"]
        self.source_nodes, self.target_nodes, self.nodes = [], [], 0
        self.collect(before["tree"], self.source_nodes)
        self.collect(after["tree"], self.target_nodes)
        if type(max_nodes) is not int or not 0 < self.nodes <= max_nodes:
            raise ModelRefusal("complete module structural node budget exceeded")
        for record in [*self.types.values(), *self.target_types.values()]:
            kind(record)
        self.changed, self.bound_targets, self.calls, self.sites = 0, set(), [], []
        self.call_targets = {_key(row["operation"]): row for row in before["calls"]}
        self.prototypes = {row["name"]: row for row in before["functions"]}
        if len(self.prototypes) != len(before["functions"]):
            raise ModelRefusal("duplicate actual function declaration")

    def collect(self, op, result):
        _metadata(op)
        self.nodes += 1
        result.append(op)
        for region in op["regions"]:
            for block in region:
                if set(block) != {"arguments", "operations"}:
                    raise ModelRefusal("unknown complete block fields")
                for child in block["operations"]:
                    self.collect(child, result)

    @staticmethod
    def operands(source, target, env):
        if len(source) != len(target) or any(env.get(a) != b for a, b in zip(source, target, strict=True)):
            raise ModelRefusal("actual operand/capture/return/observer or same-shaped buffer changed")

    def bind(self, source, target, env, context, *, known=None):
        if _types(source) != _types(target):
            raise ModelRefusal("actual SSA argument/result type or arity changed")
        result = []
        for (name, _), (candidate, _) in zip(source, target, strict=True):
            if name in context or candidate in self.bound_targets:
                raise ModelRefusal("source/candidate SSA definition rebound or capture scope violated")
            env[name] = candidate
            row = binding(name, self.types, known)
            context[name] = row
            self.bound_targets.add(candidate)
            result.append(row)
        return result

    @staticmethod
    def read(names, context):
        try:
            return [context[name] for name in names]
        except KeyError as error:
            raise ModelRefusal("actual use before scoped SSA definition") from error

    def block(self, source, target, env, context, *, terminal):
        if not source["operations"] or not target["operations"]:
            raise ModelRefusal("complete nonempty block and terminator required")
        self.bind(source["arguments"], target["arguments"], env, context)
        commands, cursor = [], 0
        for position, op in enumerate(source["operations"]):
            if position == len(source["operations"]) - 1:
                if cursor != len(target["operations"]) - 1:
                    raise ModelRefusal("extra/missing actual operation at whole-block boundary")
                last = target["operations"][cursor]
                standard_yield(op, terminal)
                standard_yield(last, terminal)
                if _metadata(op) != _metadata(last):
                    raise ModelRefusal("actual terminator metadata changed")
                self.operands(op["operands"], last["operands"], env)
                returned = self.read(op["operands"], context)
                code = dict(tag="returned", values=returned)
                break
            candidate = target["operations"][cursor] if cursor < len(target["operations"]) else None
            if op["name"] == "memref.cast":
                if (
                    op["attributes"]
                    or op["properties"]
                    or op["regions"]
                    or len(op["operands"]) != 1
                    or len(op["results"]) != 1
                ):
                    raise ModelRefusal("unsupported identity-cast metadata/shape")
                origin = self.read(op["operands"], context)[0]
                actual_source = self.types[str(origin["name"])]
                actual_result = self.types[str(op["results"][0][0])]
                if origin["kind"]["tag"] != "view" or actual_source != actual_result:
                    raise ModelRefusal("only exactly identical actual view types may be erased")
                name = op["results"][0][0]
                if name in context:
                    raise ModelRefusal("elided cast rebound an existing SSA value")
                result = binding(name, self.types)
                context[name] = result
                env[name] = env[origin["name"]]
                commands.append(
                    dict(
                        tag="cast",
                        origin=origin,
                        result=result,
                        source=origin["kind"]["type"],
                        target=result["kind"]["type"],
                    )
                )
                self.changed += 1
                self.sites.append(
                    dict(kind="cast", source_operation=_key(op), elided_result=name, candidate_value=env[name])
                )
                continue
            if candidate is None:
                raise ModelRefusal("actual candidate operation missing")
            if op["name"] == "memref.dim":
                if (
                    op["attributes"]
                    or op["properties"]
                    or op["regions"]
                    or len(op["operands"]) != 2
                    or len(op["results"]) != 1
                ):
                    raise ModelRefusal("unsupported dimension metadata/arity")
                view, axis = self.read(op["operands"], context)
                if view["kind"]["tag"] != "view" or axis["known"] is None:
                    raise ModelRefusal("whole boundary requires a statically defined dimension axis")
                index, shape = axis["known"], view["kind"]["type"]["shape"]
                if not 0 <= index < len(shape):
                    raise ModelRefusal("dimension axis outside declared shape")
                actual = index_constant(candidate, self.target_types)
                result = self.bind(op["results"], candidate["results"], env, context, known=shape[index])[0]
                commands.append(
                    dict(
                        tag="dimension",
                        view=view,
                        axis=axis,
                        result=result,
                        type=view["kind"]["type"],
                        axisIndex=index,
                        value=actual,
                    )
                )
                self.changed += 1
                self.sites.append(
                    dict(
                        kind="dimension",
                        source_operation=_key(op),
                        candidate_operation=_key(candidate),
                        expected_value=shape[index],
                        candidate_value=actual,
                    )
                )
            else:
                if _metadata(op) != _metadata(candidate):
                    raise ModelRefusal("actual retained operation/attribute/property/observer changed")
                self.operands(op["operands"], candidate["operands"], env)
                operands = self.read(op["operands"], context)
                if op["name"] == "arith.constant" and op["index_constant"] is not None:
                    value = index_constant(op, self.types)
                    index_constant(candidate, self.target_types)
                    result = self.bind(op["results"], candidate["results"], env, context, known=value)[0]
                    commands.append(dict(tag="constant", result=result, value=value))
                elif op["regions"]:
                    if op["name"] != "scf.for" or op["attributes"] or op["properties"] or len(op["operands"]) < 3:
                        raise ModelRefusal("only explicit single-body SCF regions are admitted")
                    if (
                        any(item["kind"] != dict(tag="index") for item in operands[:3])
                        or any(item["known"] is None for item in operands[:3])
                        or operands[2]["known"] <= 0
                    ):
                        raise ModelRefusal("SCF requires static nonnegative controls and positive stride")
                    if len(candidate["regions"]) != 1:
                        raise ModelRefusal("actual SCF region count changed")
                    first, second = _one_block(op), _one_block(candidate)
                    if _types(first["arguments"]) != ["index", *_types(op["results"])]:
                        raise ModelRefusal("actual SCF argument/carry types unsupported")
                    parameters = [binding(name, self.types) for name, _ in first["arguments"]]
                    body = self.block(first, second, dict(env), dict(context), terminal="scf.yield")
                    results = self.bind(op["results"], candidate["results"], env, context)
                    commands.append(
                        dict(
                            tag="region",
                            key=operation_key(op),
                            operands=operands,
                            parameters=parameters,
                            results=results,
                            body=body,
                        )
                    )
                else:
                    if candidate["regions"]:
                        raise ModelRefusal("actual retained operation gained hidden control/effects")
                    if op["name"] in ("memref.subview", "memref.reinterpret_cast", "memref.view", "memref.reshape"):
                        raise ModelRefusal("unknown descriptor/alias construction is outside this admitted boundary")
                    results = self.bind(op["results"], candidate["results"], env, context)
                    key = operation_key(op)
                    commands.append(dict(tag="action", key=key, operands=operands, results=results))
                    if op["name"] == "func.call":
                        call = self.call_targets.get(_key(op))
                        if call is None or call["nested"]:
                            raise ModelRefusal("actual call target/declaration unavailable")
                        self.calls.append([key, call["target"]])
            cursor += 1
        for command in reversed(commands):
            code = dict(command, rest=code)
        return code

    def project(self, *, allow_identity=False):
        source, target = self.before["tree"], self.after["tree"]
        if (
            source["name"] != "builtin.module"
            or _metadata(source) != _metadata(target)
            or any(source[field] for field in ("operands", "results", "attributes", "properties"))
        ):
            raise ModelRefusal("complete plain module boundary required")
        first, second = _one_block(source), _one_block(target)
        if first["arguments"] or second["arguments"] or len(first["operations"]) != len(second["operations"]):
            raise ModelRefusal("actual module scope/function count changed")
        functions = []
        for op, candidate, prototype in zip(
            first["operations"], second["operations"], self.before["functions"], strict=True
        ):
            if op["name"] != "func.func" or _metadata(op) != _metadata(candidate) or op["operands"] or op["results"]:
                raise ModelRefusal("actual complete function/signature/visibility changed")
            for record in prototype["parameters"] + prototype["returned"]:
                kind(record)
            if prototype["declaration"]:
                if op != candidate:
                    raise ModelRefusal("external declaration changed")
                continue
            body, after_body = _one_block(op), _one_block(candidate)
            parameters = [binding(name, self.types) for name, _ in body["arguments"]]
            code = self.block(body, after_body, {}, {}, terminal="func.return")
            functions.append(
                dict(
                    name=prototype["name"],
                    parameters=parameters,
                    returned=[kind(record) for record in prototype["returned"]],
                    code=code,
                )
            )
        if not self.changed and (not allow_identity or source != target):
            raise ModelRefusal("identity/declined normalization is separate from changed qualification")
        result = dict(
            bits=self.bits,
            module=dict(functions=functions, calls=self.calls),
            changed_sites=self.sites,
            structural_nodes=self.nodes,
            source_binding=_key(source),
            target_binding=_key(target),
            typed_descriptor_conformance_is_premise=True,
            python_projection_trusted=True,
        )
        if allow_identity and not self.changed:
            result["identity_normalization"] = True
        return result


def project_exports(before, after, *, index_bits, max_nodes):
    return Decoder(before, after, index_bits=index_bits, max_nodes=max_nodes).project()
