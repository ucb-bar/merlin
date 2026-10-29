"""Bind the scoped shape/epoch adapter to complete actual normalized syntax.

The frozen shape matcher supplies scoped retained-SSA pairs. The normalized
candidate is independently decoded as ordinary Code, then alpha-renamed using
those pairs. Lean compares that complete Code with its proved normalizer. No
identity producer run or assumed whole-program semantic join is introduced.
"""

from __future__ import annotations

import copy
import json

from .lean_graph_loops_segment_projection import _one_block
from .lean_graph_loops_shape_projection import Decoder
from .lean_graph_loops_shape_types import binding, kind


class BoundDecoder(Decoder):
    def __init__(self, before, after, *, index_bits, max_nodes):
        self.candidate_to_source = {}
        super().__init__(before, after, index_bits=index_bits, max_nodes=max_nodes)

    def bind(self, source, target, env, context, *, known=None):
        result = super().bind(source, target, env, context, known=known)
        for original, candidate in zip(source, target, strict=True):
            if candidate[0] in self.candidate_to_source:
                raise ValueError("actual retained SSA pair was recorded twice")
            self.candidate_to_source[candidate[0]] = original[0]
        return result


def plain_module(exported, *, index_bits, max_nodes):
    """Decode existing candidate syntax, with no claim that identity IR changed."""
    decoder = Decoder(exported, exported, index_bits=index_bits, max_nodes=max_nodes)
    if any(op["name"] in ("memref.cast", "memref.dim") for op in decoder.source_nodes):
        raise ValueError("normalized candidate still contains a cast/dimension site")
    functions = []
    operations = _one_block(exported["tree"])["operations"]
    for op, prototype in zip(operations, exported["functions"], strict=True):
        if op["name"] != "func.func":
            raise ValueError("complete plain candidate function table required")
        if prototype["declaration"]:
            continue
        body = _one_block(op)
        parameters = [binding(name, decoder.types) for name, _ in body["arguments"]]
        code = decoder.block(body, body, {}, {}, terminal="func.return")
        functions.append(
            dict(
                name=prototype["name"],
                parameters=parameters,
                returned=[kind(record) for record in prototype["returned"]],
                code=code,
            )
        )
    return dict(functions=functions, calls=decoder.calls)


def rename_bindings(value, candidate_to_source):
    if isinstance(value, dict):
        if set(value) == {"name", "kind", "known"}:
            return dict(value, name=candidate_to_source[value["name"]])
        return {key: rename_bindings(child, candidate_to_source) for key, child in value.items()}
    if isinstance(value, list):
        return [rename_bindings(child, candidate_to_source) for child in value]
    return value


def top_regions(code):
    regions = []
    while code["tag"] != "returned":
        if code["tag"] == "region":
            regions.append(dict(code, rest=dict(tag="returned", values=[])))
        code = code["rest"]
    return regions


def uniform(region):
    if region["tag"] != "region" or region["results"] or len(region["parameters"]) != 1:
        raise ValueError("partial epoch adapter requires one outer induction variable and no carries")
    inner = region["body"]
    if inner["tag"] != "region" or inner["results"] or len(inner["parameters"]) != 1:
        raise ValueError("partial epoch adapter requires exactly two nested loops")
    cell = inner["body"]
    if cell["tag"] != "action" or cell["results"] or json.loads(cell["key"])["name"] != "memref.store":
        raise ValueError("partial epoch adapter requires one concrete store body")
    if any(row["rest"] != dict(tag="returned", values=[]) for row in (region, inner, cell)):
        raise ValueError("initializer gained a surrounding/leaf effect")
    if len(region["operands"]) != 3 or len(inner["operands"]) != 3 or len(cell["operands"]) != 4:
        raise ValueError("initializer control/store operand arity changed")
    zero, rows, one = region["operands"]
    inner_zero, columns, inner_one = inner["operands"]
    outer_iv, inner_iv = region["parameters"][0], inner["parameters"][0]
    seed, buffer, row, column = cell["operands"]
    if (zero, one, row, column) != (inner_zero, inner_one, outer_iv, inner_iv):
        raise ValueError("initializer controls or actual store coordinates differ")
    if region["key"] != inner["key"] or json.loads(region["key"])["name"] != "scf.for":
        raise ValueError("initializer must use the same plain SCF operation meaning")
    return dict(
        storeKey=cell["key"],
        loopKey=region["key"],
        seed=seed,
        buffer=buffer,
        outer=outer_iv,
        inner=inner_iv,
        zero=zero,
        rows=rows,
        columns=columns,
        one=one,
    )


def project_adapter(before, after, *, index_bits, max_nodes):
    decoder = BoundDecoder(before, after, index_bits=index_bits, max_nodes=max_nodes)
    source = decoder.project(allow_identity=True)
    plain = plain_module(after, index_bits=index_bits, max_nodes=max_nodes)
    renamed = rename_bindings(plain, decoder.candidate_to_source)
    initializers = []
    for function in renamed["functions"]:
        for position, region in enumerate(top_regions(function["code"])):
            program = uniform(region)
            view_type = program["buffer"]["kind"]
            if view_type["tag"] != "view" or len(view_type["type"]["shape"]) != 2:
                raise ValueError("partial initializer adapter requires a rank-two view")
            initializers.append(
                dict(
                    function=function["name"],
                    position=position,
                    program=program,
                    shape=view_type["type"]["shape"],
                    element=view_type["type"]["element"],
                )
            )
    return dict(
        shape=source,
        candidate_plain=plain,
        normalized_candidate=renamed,
        candidate_to_source=copy.deepcopy(decoder.candidate_to_source),
        initializers=initializers,
        semantic_validation_pending=True,
        python_projection_trusted=True,
    )


def bind_memory(adapter, normalized_export, memory_exports, *, index_bits, max_nodes):
    """Reproject the actual triple and link every initializer to its raw SSA body.

    The entire normalized tree is compared before using any epoch configuration.
    All old projection checks run afresh as pure host checks; no saved model or
    receipt alone establishes this new operational connection.
    """
    from .lean_graph_loops_memory_scf_projection import project_exports
    from .lean_graph_loops_segment_projection import _key

    if normalized_export["tree"] != memory_exports["before_export"]["tree"]:
        raise ValueError("complete normalized module differs from the memory-edge source")
    memory = project_exports(
        memory_exports["before_export"],
        memory_exports["middle_export"],
        memory_exports["after_export"],
        index_bits=index_bits,
        max_nodes=max_nodes,
    )
    configurations = {row["key"]: row for row in memory["configurations"]}
    if len(configurations) != len(memory["configurations"]):
        raise ValueError("ambiguous raw initializer configuration")
    links = []
    raw_functions = adapter["candidate_plain"]["functions"]
    operations = _one_block(normalized_export["tree"])["operations"]
    mapping = {int(raw): source for raw, source in adapter["candidate_to_source"].items()}
    for op, prototype in zip(operations, normalized_export["functions"], strict=True):
        if prototype["declaration"]:
            continue
        function = next(row for row in raw_functions if row["name"] == prototype["name"])
        regions = top_regions(function["code"])
        raw_regions = [node for node in _one_block(op)["operations"] if node["name"] == "scf.for"]
        if len(regions) != len(raw_regions):
            raise ValueError("unmapped complete top-level region")
        for position, (raw_op, region) in enumerate(zip(raw_regions, regions, strict=True)):
            key = _key(raw_op)
            if key not in configurations:
                raise ValueError("actual typed initializer has no checked old configuration")
            config = configurations.pop(key)
            epoch = memory["reuse"]["epochs"][config["object"]]
            program = uniform(region)
            canonical = rename_bindings(program, mapping)
            expected = [
                row
                for row in adapter["initializers"]
                if row["function"] == function["name"] and row["position"] == position
            ]
            if len(expected) != 1 or expected[0]["program"] != canonical:
                raise ValueError("canonical initializer lost its complete actual syntax binding")
            if program["buffer"]["name"] != epoch["source_value"]:
                raise ValueError("initializer store names the wrong private allocation")
            if config["program"] != dict(
                shape=expected[0]["shape"],
                ivs=[program["outer"]["name"], program["inner"]["name"]],
                outerPrefix=[],
                leaf=dict(
                    ops=[], value=program["seed"]["name"], indices=[program["outer"]["name"], program["inner"]["name"]]
                ),
            ):
                raise ValueError("raw epoch scalar/coordinate program differs from the typed initializer")
            links.append(
                dict(
                    raw=program,
                    canonical=canonical,
                    config=config,
                    mapping=sorted(mapping.items()),
                    rows=expected[0]["shape"][0],
                    columns=expected[0]["shape"][1],
                    element=expected[0]["element"],
                    sourceBuffer=epoch["source_value"],
                    function=function["name"],
                    position=position,
                )
            )
    if configurations or len(links) != len(memory["configurations"]):
        raise ValueError("unbound initializer configuration at complete memory boundary")
    return dict(adapter, memory=memory, initializer_links=links, full_epoch_public_adapter_proved=False)
