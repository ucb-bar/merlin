"""Independently decode each actual shape boundary into ordinary typed Code.

No source/candidate full-tree matcher supplies the final body. The two decodes
are independent; a finite bijective renaming is only a hint for the Lean pair
check. In-domain wrong final programs must produce a different, failing query.
Parsing and input admission belong to the shared capture owner.
"""

from __future__ import annotations

from .lean_graph_loops_segment_projection import _key, _metadata, _one_block, _types
from .lean_graph_loops_shape_projection import Decoder
from .lean_graph_loops_shape_types import binding, index_constant, kind, operation_key, standard_yield


class SingleDecoder(Decoder):
    """Use shared scalar/type helpers, never the old pair-matching block method."""

    def __init__(self, exported, *, index_bits, max_nodes):
        super().__init__(exported, exported, index_bits=index_bits, max_nodes=max_nodes)
        self.frame = []

    def define(self, values, context, *, known=None):
        return self.bind(values, values, {}, context, known=known)

    def check_loop_controls(self, operands):
        if (
            any(row["kind"] != dict(tag="index") or row["known"] is None for row in operands[:3])
            or operands[2]["known"] <= 0
        ):
            raise ValueError("ordinary SCF requires static controls and positive stride")

    def ordinary_block(self, block, context, *, terminal):
        if not block["operations"]:
            raise ValueError("complete ordinary block requires a terminator")
        self.define(block["arguments"], context)
        commands = []
        for position, op in enumerate(block["operations"]):
            if position == len(block["operations"]) - 1:
                standard_yield(op, terminal)
                code = dict(tag="returned", values=self.read(op["operands"], context))
                break
            operands = self.read(op["operands"], context)
            name = op["name"]
            if name == "memref.cast":
                if (
                    op["attributes"]
                    or op["properties"]
                    or op["regions"]
                    or len(operands) != 1
                    or len(op["results"]) != 1
                ):
                    raise ValueError("ordinary cast requires exact supported metadata and arity")
                origin = operands[0]
                result = self.define(op["results"], context)[0]
                if origin["kind"]["tag"] != "view" or result["kind"]["tag"] != "view":
                    raise ValueError("cast requires ranked descriptor types")
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
            elif name == "memref.dim":
                if (
                    op["attributes"]
                    or op["properties"]
                    or op["regions"]
                    or len(operands) != 2
                    or len(op["results"]) != 1
                ):
                    raise ValueError("ordinary dimension requires exact metadata and arity")
                view, axis = operands
                if view["kind"]["tag"] != "view" or axis["known"] is None:
                    raise ValueError("dimension requires a ranked view and known axis")
                index, shape = axis["known"], view["kind"]["type"]["shape"]
                if not 0 <= index < len(shape):
                    raise ValueError("dimension axis outside its declared shape")
                value = shape[index]
                result = self.define(op["results"], context, known=value)[0]
                commands.append(
                    dict(
                        tag="dimension",
                        view=view,
                        axis=axis,
                        result=result,
                        type=view["kind"]["type"],
                        axisIndex=index,
                        value=value,
                    )
                )
                self.changed += 1
            elif name == "arith.constant" and op["index_constant"] is not None:
                value = index_constant(op, self.types)
                result = self.define(op["results"], context, known=value)[0]
                commands.append(dict(tag="constant", result=result, value=value))
            elif op["regions"]:
                if name != "scf.for" or op["attributes"] or op["properties"] or len(operands) < 3:
                    raise ValueError("ordinary region outside the single-body SCF subset")
                self.check_loop_controls(operands)
                body = _one_block(op)
                if _types(body["arguments"]) != ["index", *_types(op["results"])]:
                    raise ValueError("ordinary SCF induction/carry signature differs")
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
                if name in ("memref.subview", "memref.reinterpret_cast", "memref.view", "memref.reshape"):
                    raise ValueError("descriptor construction outside the ordinary shape subset")
                results = self.define(op["results"], context)
                key = operation_key(op)
                commands.append(dict(tag="action", key=key, operands=operands, results=results))
                if name == "func.call":
                    call = self.call_targets.get(_key(op))
                    if call is None or call["nested"]:
                        raise ValueError("ordinary call has no admitted flat target")
                    self.calls.append([key, call["target"]])
        for command in reversed(commands):
            code = dict(command, rest=code)
        return code

    def ordinary_module(self):
        tree = self.before["tree"]
        if tree["name"] != "builtin.module" or any(
            tree[field] for field in ("operands", "results", "attributes", "properties")
        ):
            raise ValueError("ordinary whole plain module required")
        block = _one_block(tree)
        if block["arguments"]:
            raise ValueError("module block arguments are outside this subset")
        self.frame.append(_key(_metadata(tree)))
        functions = []
        for op, prototype in zip(block["operations"], self.before["functions"], strict=True):
            if op["name"] != "func.func" or op["operands"] or op["results"]:
                raise ValueError("every module operation must be a complete function")
            for record in prototype["parameters"] + prototype["returned"]:
                kind(record)
            self.frame.append(_key(dict(metadata=_metadata(op), prototype=prototype)))
            if prototype["declaration"]:
                if len(op["regions"]) != 1 or op["regions"][0]:
                    raise ValueError("declaration unexpectedly contains a region")
                continue
            body = _one_block(op)
            parameters = [binding(identity, self.types) for identity, _ in body["arguments"]]
            functions.append(
                dict(
                    name=prototype["name"],
                    parameters=parameters,
                    returned=[kind(row) for row in prototype["returned"]],
                    code=self.ordinary_block(body, {}, terminal="func.return"),
                )
            )
        return dict(functions=functions, calls=self.calls)


def decode(exported, *, index_bits, max_nodes):
    decoder = SingleDecoder(exported, index_bits=index_bits, max_nodes=max_nodes)
    module = decoder.ordinary_module()
    return dict(module=module, frame=decoder.frame, shape_sites=decoder.changed, structural_nodes=decoder.nodes)


def _renamed(binding_value, names):
    return dict(binding_value, name=names.get(binding_value["name"], binding_value["name"]))


def normalized_code(code, aliases):
    """Only constructs an SSA-permutation hint; never emitted as actual after."""
    tag = code["tag"]
    if tag == "returned":
        return dict(tag=tag, values=[_renamed(row, aliases) for row in code["values"]])
    if tag == "cast":
        target = aliases.get(code["origin"]["name"], code["origin"]["name"])
        return normalized_code(code["rest"], {**aliases, code["result"]["name"]: target})
    if tag in ("dimension", "constant"):
        result = code["result"]
        return dict(
            tag="constant",
            result=result,
            value=code["value"],
            rest=normalized_code(code["rest"], {**aliases, result["name"]: result["name"]}),
        )
    results = code["results"]
    suffix = {**aliases, **{row["name"]: row["name"] for row in results}}
    node = dict(
        tag=tag,
        key=code["key"],
        operands=[_renamed(row, aliases) for row in code["operands"]],
        results=results,
        rest=normalized_code(code["rest"], suffix),
    )
    if tag == "region":
        parameters = code["parameters"]
        node.update(
            parameters=parameters,
            body=normalized_code(code["body"], {**aliases, **{row["name"]: row["name"] for row in parameters}}),
        )
    return node


def definitions(code):
    tag = code["tag"]
    if tag == "returned":
        return []
    if tag in ("constant", "cast", "dimension"):
        return [code["result"]["name"], *definitions(code["rest"])]
    nested = [] if tag == "action" else [*[row["name"] for row in code["parameters"]], *definitions(code["body"])]
    return [*nested, *[row["name"] for row in code["results"]], *definitions(code["rest"])]


def definition_roster(module):
    return [
        identity
        for function in module["functions"]
        for identity in [*[row["name"] for row in function["parameters"]], *definitions(function["code"])]
    ]


def permutation_hint(before, after):
    normalized = dict(
        before,
        functions=[dict(function, code=normalized_code(function["code"], {})) for function in before["functions"]],
    )
    actual, expected = definition_roster(after), definition_roster(normalized)
    swaps = []
    for source, target in zip(actual, expected):
        current = source
        for left, right in reversed(swaps):
            current = right if current == left else left if current == right else current
        if current != target:
            swaps.insert(0, [current, target])
    return swaps


def project_pair(before, after, *, index_bits, max_nodes):
    source = decode(before, index_bits=index_bits, max_nodes=max_nodes)
    target = decode(after, index_bits=index_bits, max_nodes=max_nodes)
    return dict(
        bits=index_bits,
        before=source["module"],
        after=target["module"],
        swaps=permutation_hint(source["module"], target["module"]),
        beforeFrame=source["frame"],
        afterFrame=target["frame"],
        beforeBinding=_key(before["tree"]),
        afterBinding=_key(after["tree"]),
        independent_after_term=True,
        host_pair_match_used=False,
        lean_checked=False,
    )
