"""SOURCE-ONLY strict retained-export/Kernel/plan to Lean literal association.

This is a named trusted Python projection, not a Lean theorem or qualification.
It uses only retained source/JSON and the unchanged v1 serializer source. No
SDK/binary, compiler, provider, loader, proof or MLIR invocation is permitted.
"""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

SERIALIZER = (5430, "f14077f533c73685d10ec716951529c8098f158613a5383d141db64ea070db5b")
CANDIDATE = (32486, "72e002204f9c474b0086153739bbc0828c7f7f24bd33d51338f07a72890770ad")


def fields(value, names):
    if type(value) is not dict or set(value) != set(names.split()):
        raise ValueError("complete exact structural fields required: " + names)


def each(values, visit):
    if type(values) is not list:
        raise ValueError("structural list required")
    for value in values:
        visit(value)


def metadata(values):
    if type(values) is not dict or any(type(key) is not str for key in values):
        raise ValueError("complete metadata dictionary required")
    for value in values.values():
        attribute(value)


def type_schema(value):
    kind = value.get("kind") if type(value) is dict else None
    simple = {"int": "width", "sint": "width", "uint": "width", "index": "", "none": "",
              "float": "format", "opaque": "text"}
    if kind in simple:
        fields(value, "kind " + simple[kind])
    elif kind == "tensor":
        fields(value, "kind shape element encoding")
        type_schema(value["element"])
        if value["encoding"] is not None:
            attribute(value["encoding"])
    elif kind == "function":
        fields(value, "kind inputs results")
        each(value["inputs"], type_schema)
        each(value["results"], type_schema)
    else:
        raise ValueError("unsupported structural type")


def affine(value):
    kind = value.get("kind") if type(value) is dict else None
    if kind in {"dim", "symbol"}:
        fields(value, "kind pos")
    elif kind == "constant":
        fields(value, "kind value")
    elif kind in {"add", "mul", "mod", "floordiv", "ceildiv"}:
        fields(value, "kind lhs rhs")
        affine(value["lhs"])
        affine(value["rhs"])
    else:
        raise ValueError("unsupported complete affine syntax")


def attribute(value):
    kind = value.get("kind") if type(value) is dict else None
    simple = {"unit": "", "bool": "value", "string": "value", "enum": "dialect name value",
              "symbol_ref": "root nested", "opaque": "text"}
    if kind in simple:
        fields(value, "kind " + simple[kind])
    elif kind in {"int", "float"}:
        fields(value, "kind type " + ("value" if kind == "int" else "bits"))
        type_schema(value["type"])
    elif kind == "type":
        fields(value, "kind value")
        type_schema(value["value"])
    elif kind == "array":
        fields(value, "kind elements")
        each(value["elements"], attribute)
    elif kind == "dense_array":
        fields(value, "kind element values")
        type_schema(value["element"])
    elif kind == "dense":
        fields(value, "kind type splat hex")
        type_schema(value["type"])
    elif kind == "affine_map":
        fields(value, "kind dims symbols results")
        each(value["results"], affine)
    elif kind == "dict":
        fields(value, "kind entries")
        metadata(value["entries"])
    else:
        raise ValueError("unsupported complete attribute syntax")


def variable(value):
    fields(value, "id type")
    type_schema(value["type"])


def block(value):
    fields(value, "args ops")
    each(value["args"], variable)
    each(value["ops"], operation)


def region(value):
    fields(value, "blocks")
    each(value["blocks"], block)


def operation(value):
    fields(value, "name results operands attrs props regions successors")
    each(value["results"], variable)
    metadata(value["attrs"])
    metadata(value["props"])
    each(value["regions"], region)


def document_schema(value):
    fields(value, "schema producer source module")
    fields(value["producer"], "tool version")
    fields(value["source"], "bytes sha256")
    fields(value["module"], "ops attrs props")
    each(value["module"]["ops"], operation)
    metadata(value["module"]["attrs"])
    metadata(value["module"]["props"])


def module_term(document, serializer):
    document_schema(document)
    return serializer["module"](document)


def require_literal(source, name, kind, term):
    """Canonical one-line literal protocol, after complete source pin checking.

    This is not a parser for arbitrary Lean/comments/namespaces. `associate`
    first binds the entire known generated source, then checks its actual IR
    literal bytes. Native proof checking remains held.
    """
    expected = f"def {name} : {kind} := {term}"
    if source.splitlines().count(expected) != 1:
        raise ValueError("exact complete generated literal association unavailable: " + name)
    return dict(name=name, type=kind, exact_complete_declaration=expected)


def raw_kernel(value, serializer):
    fields(value, "name args attrs body")
    a, t, n, optional = (serializer[key] for key in ("array", "text", "natural", "optional"))

    def pairs(values, encode):
        return a(values, lambda pair: f"({t(pair[0])},{encode(pair[1])})")

    def argument(value):
        fields(value, "name shape dtype access")
        return f"⟨{t(value['name'])},{a(value['shape'], n)},{t(value['dtype'])},{t(value['access'])}⟩"

    def call(value):
        fields(value, "instr args assumes awaits produces sets stages unit")
        arguments = []
        for key, expression in value["args"]:
            fields(expression, "value")
            arguments.append((key, expression["value"]))
        return (
            f"⟨{t(value['instr'])},{pairs(arguments, n)},{a(value['assumes'], t)},{a(value['awaits'], t)},"
            f"{optional(value['produces'], t)},{a(value['sets'], t)},{a(value['stages'], t)},"
            f"{optional(value['unit'], t)}⟩"
        )

    return f"⟨{t(value['name'])},{a(value['args'], argument)},{pairs(value['attrs'], t)},{a(value['body'], call)}⟩"


def source_locators(document):
    """Pinned API walk includes builtin.module ordinal0, absent from export.

    Walk1 is the exported func. Block arguments are owned by their enclosing op;
    ambiguous multi-region/block argument identities are explicitly refused.
    This is a host locator convention, not a theorem of the MLIR parser.
    """
    values, nodes = {}, {}
    ordinal = 0

    def bind(key, value):
        if key in values:
            raise ValueError("ambiguous source value locator")
        values[key] = value

    def visit(op, path, body_index):
        nonlocal ordinal
        ordinal += 1
        owner = ordinal
        nodes[owner] = dict(path=path, body_index=body_index, name=op["name"])
        for index, result in enumerate(op["results"]):
            bind(("result", owner, index), result["id"])
        for ri, reg in enumerate(op["regions"]):
            for bi, blk in enumerate(reg["blocks"]):
                for index, arg in enumerate(blk["args"]):
                    bind(("argument", owner, index), arg["id"])
                for index, nested in enumerate(blk["ops"]):
                    visit(nested, f"{path}/regions/{ri}/blocks/{bi}/ops/{index}", index)

    for index, op in enumerate(document["module"]["ops"]):
        visit(op, f"/module/ops/{index}", index)
    return values, nodes


def bind_plan(plan, locators, source, serializer):
    fields(plan, "inputs maturity outputs source_ir_sha256 stages values")
    values, nodes = source_locators(source)
    resolved = []
    for locator in locators:
        fields(locator, "index kind owner_ordinal type")
        key = (locator["kind"], locator["owner_ordinal"], locator["index"])
        if key not in values:
            raise ValueError("missing actual source value locator")
        resolved.append(values[key])
    if len(resolved) != len(plan["values"]):
        raise ValueError("plan values/locators differ")
    stages, rows = [], []
    for index, stage in enumerate(plan["stages"]):
        fields(stage, "inputs kind ordinal output")
        node = nodes[stage["ordinal"]]
        if node["name"] != "linalg.generic" or stage["kind"] != "elementwise_map":
            raise ValueError("selected plan does not resolve to an actual generic")
        rows.append(dict(stage=index, producer_walk=stage["ordinal"], **node))
        stages.append(
            f"⟨stage{index}.ordinal,stage{index}.role,{serializer['array'](stage['inputs'], serializer['natural'])},"
            f"[{serializer['natural'](stage['output'])}],stage{index}.inputTypes,[stage{index}.outputType]⟩"
        )
    a, n = serializer["array"], serializer["natural"]
    lifetimes = []
    for value in plan["values"]:
        fields(value, "type lifetime")
        if type(value["lifetime"]) is not list or len(value["lifetime"]) != 2:
            raise ValueError("complete lifetime pair required")
        first, last = value["lifetime"]
        if type(first) is not int:
            raise ValueError("strict signed lifetime required")
        lifetimes.append(f"({first},{n(last)})")
    term = (
        f"⟨{a(resolved, n)},[{','.join(stages)}],{a(plan['inputs'], n)},"
        f"{a(plan['outputs'], n)},[{','.join(lifetimes)}]⟩"
    )
    return term, dict(actual_value_ids=resolved, stage_rows=rows)


def associate(root, helper, destination):
    api = runpy.run_path(str(helper))
    base = root / api["BASE"]
    serializer_pin = dict(
        path=str(base / "qualification-v1/source-snapshot/src/merlin/verify/lean_graph_regions_declared_ast.py"),
        bytes=SERIALIZER[0], sha256=SERIALIZER[1],
    )
    api["read_bound"](root, serializer_pin)
    serializer = runpy.run_path(serializer_pin["path"])
    _, evidence = api["original_evidence"](
        root, "qualification-v1-originals/verified-i8/EVIDENCE.json", api["EVIDENCE"]
    )
    binding = json.loads((base / "retained-literal-binding-v3/BINDING.json").read_bytes())
    source_bytes = api["read_bound"](root, binding["generated_snapshot"])
    if api["identity"](source_bytes) != dict(bytes=CANDIDATE[0], sha256=CANDIDATE[1]):
        raise ValueError("entire current generated-source pin differs; literal lines cannot substitute a proof")
    primary = root / "src/merlin/verify/lean/Merlin/GraphRegionsDeclaredCaptureRetainedI8.lean"
    if primary.read_bytes() != source_bytes:
        raise ValueError("current candidate and retained generated source differ")
    source = source_bytes.decode()
    literals, documents = [], {}
    for role, pin in evidence["exports"].items():
        document = json.loads(api["read_bound"](root, pin))
        documents[role] = document
        original_name = {"before": "source.mlir", "retained": "retained-source.mlir"}.get(role, role + ".mlir")
        original = api["read_bound"](root, evidence["originals"][original_name])
        if original != api["read_bound"](root, evidence["snapshots"][original_name]):
            raise ValueError("actual original/snapshot bytes differ")
        if document["source"] != api["identity"](original):
            raise ValueError("export source/actual original association differs")
        name = {"before": "actualBefore", "retained": "actualRetained"}.get(role)
        if name is None:
            _, index, scalar_role = role.split("-")
            name = "scalar" + index + "_" + scalar_role
        literals.append(require_literal(source, name, "Module", module_term(document, serializer)))
    raw_roles = (("actual-output-kernel.json", "actualKernel"), ("original-provider-kernel.json", "originalKernel"))
    for filename, name in raw_roles:
        data = api["read_bound"](root, evidence["originals"][filename])
        if data != api["read_bound"](root, evidence["snapshots"][filename]):
            raise ValueError("raw original/snapshot bytes differ")
        literals.append(require_literal(source, name, "RawKernel", raw_kernel(json.loads(data), serializer)))
    plan = json.loads(api["read_bound"](root, evidence["originals"]["original-connected-plan.json"]))
    locators = json.loads(api["read_bound"](root, evidence["originals"]["original-connected-plan-values.json"]))
    if plan["source_ir_sha256"] != documents["before"]["source"]["sha256"]:
        raise ValueError("plan source differs from the retained original")
    term, mapped = bind_plan(plan, locators, documents["before"], serializer)
    literals.append(require_literal(source, "actualPlan", "Plan", term))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "Candidate.lean").write_bytes(source_bytes)
    record = dict(
        schema="merlin.graph-regions.source-only-projection-association.v1",
        status="SOURCE DATA ASSOCIATED; UNQUALIFIED; NO new Lean/MLIR/native run",
        literals=literals, planner_field_map=mapped, serializer=serializer_pin,
        candidate=dict(path=str(primary), **api["identity"](source_bytes)),
        snapshot=dict(path=str(destination / "Candidate.lean"), **api["identity"](source_bytes)),
        source_only=True, native_invocations=0, sdk_binary_reads=0, independent_review=False,
        trusted_host_projection=True, graph_native_qualification=False, selected_runtime_hold=True,
    )
    (destination / "ASSOCIATION.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--helper", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    arguments = parser.parse_args()
    record = associate(
        arguments.root.resolve(strict=True), arguments.helper.resolve(strict=True), arguments.destination
    )
    print(json.dumps(dict(
        status=record["status"], literal_count=len(record["literals"]), planner=record["planner_field_map"]
    )))
