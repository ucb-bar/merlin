"""Untrusted descriptions from actual exported ASTs; acceptance belongs to Lean.

Unsupported scalar names receive a mismatching supported template, so complete
source matching refuses them. Unknown execution is never replaced by success.
"""

from __future__ import annotations

from merlin.verify import lean_graph_regions_source_ast as ast

BINARY = {
    "arith." + name: kind
    for name, kind in (
        ("addi", "add"),
        ("subi", "sub"),
        ("muli", "mul"),
        ("andi", "band"),
        ("ori", "bor"),
        ("xori", "bxor"),
        ("minsi", "minsi"),
        ("maxsi", "maxsi"),
    )
}
CAST = {"arith.trunci": "trunc", "arith.extsi": "signed", "arith.extui": "unsigned"}


def tensor_type(value):
    if value["kind"] != "tensor" or value["encoding"] is not None or value["element"]["kind"] != "int":
        raise ValueError("source model requires an actual unencoded static integer tensor")
    for dimension in value["shape"]:
        ast.nat(dimension)
    return {"shape": value["shape"], "width": value["element"]["width"]}


def width(value):
    # A mismatching template is sufficient for unsupported scalar types; the
    # complete actual AST is separately serialized without this fallback.
    return value["width"] if value["kind"] == "int" else 8


def propose(document, keys):
    functions = document["module"]["ops"]
    if len(functions) != 1 or functions[0]["name"] != "func.func":
        raise ValueError("one actual defined source function required")
    function = functions[0]
    block = function["regions"][0]["blocks"][0]
    arguments = [[argument["id"], tensor_type(argument["type"])] for argument in block["args"]]
    signature = function["attrs"]["function_type"]["value"]
    result = dict(
        name=function["attrs"]["sym_name"]["value"],
        arguments=arguments,
        items=[],
        returned=block["ops"][-1]["operands"],
        results=[tensor_type(ty) for ty in signature["results"]],
    )
    ordinal, ordinals = 0, {}
    types, constants = {}, {}

    def number(operation):
        nonlocal ordinal
        ordinal += 1  # builtin.module occupies ordinal zero in the production walk
        ordinals[id(operation)] = ordinal
        for variable in operation["results"]:
            types[variable["id"]] = variable["type"]
        for region in operation["regions"]:
            for region_block in region["blocks"]:
                for variable in region_block["args"]:
                    types[variable["id"]] = variable["type"]
                for nested in region_block["ops"]:
                    number(nested)

    number(function)

    def constant(operation):
        variable = operation["results"][0]
        value = operation["attrs"].get("value", {})
        literal = int(value["value"]) if value.get("kind") == "int" else 0
        return dict(id=variable["id"], width=width(variable["type"]), literal=literal)

    def instruction(operation):
        variable = operation["results"][0]
        w = width(variable["type"])
        name = operation["name"]
        if name == "arith.constant":
            code = dict(kind="constant", width=w, literal=constant(operation)["literal"])
        elif name in CAST:
            code = dict(kind="cast", input=width(types[operation["operands"][0]]), output=w, operator=CAST[name])
        else:
            code = dict(kind="binary", width=w, operator=BINARY.get(name, "add"))
        return dict(code=code, result=variable["id"], operands=operation["operands"])

    for operation in block["ops"][:-1]:
        name = operation["name"]
        if name == "tensor.empty":
            variable = operation["results"][0]
            result["items"].append(dict(kind="empty", result=variable["id"], type=tensor_type(variable["type"])))
        elif name == "arith.constant":
            captured = constant(operation)
            constants[captured["id"]] = captured
            result["items"].append(dict(kind="constant", value=captured))
        elif name == "linalg.generic":
            variable = operation["results"][0]
            output = tensor_type(variable["type"])
            body = operation["regions"][0]["blocks"][0]
            body_ids = {value["id"] for value in body["args"]}
            body_ids.update(value["id"] for op in body["ops"] for value in op["results"])
            captured_ids = list(
                dict.fromkeys(
                    value
                    for op in body["ops"]
                    for value in op["operands"]
                    if value not in body_ids and value in constants
                )
            )
            body_description = dict(
                inputs=[[value["id"], width(value["type"])] for value in body["args"][:-1]],
                old=body["args"][-1]["id"],
                width=output["width"],
                instructions=[instruction(op) for op in body["ops"][:-1]],
                returned=body["ops"][-1]["operands"][0],
                captures=[constants[value] for value in captured_ids],
            )
            stage = dict(
                ordinal=ordinals[id(operation)],
                inputs=operation["operands"][:-1],
                destination=operation["operands"][-1],
                result=variable["id"],
                shape=output["shape"],
                body=body_description,
            )
            result["items"].append(dict(kind="generic", stage=stage, key=keys[stage["ordinal"]]))
        else:
            raise ValueError("surrounding source operation outside current model")
    return result


def type_term(value):
    return f"⟨{ast.sequence(value['shape'], ast.nat)},{ast.nat(value['width'])}⟩"


def constant_term(value):
    return f"⟨{ast.nat(value['id'])},{ast.nat(value['width'])},{ast.integer(value['literal'])}⟩"


def code_term(value):
    kind = value["kind"]
    if kind == "constant":
        return f"(.constant {ast.nat(value['width'])} {ast.integer(value['literal'])})"
    if kind == "binary" and value["operator"] in BINARY.values():
        return f"(.binary {ast.nat(value['width'])} .{value['operator']})"
    if kind == "cast" and value["operator"] in CAST.values():
        return f"(.cast {ast.nat(value['input'])} {ast.nat(value['output'])} .{value['operator']})"
    raise ValueError("unknown model code")


def instruction_term(value):
    return f"⟨{code_term(value['code'])},{ast.nat(value['result'])},{ast.sequence(value['operands'], ast.nat)}⟩"


def body_term(value):
    pairs = ast.sequence(value["inputs"], lambda pair: f"({ast.nat(pair[0])},{ast.nat(pair[1])})")
    return (
        f"⟨{pairs},{ast.nat(value['old'])},{ast.nat(value['width'])},"
        f"{ast.sequence(value['instructions'], instruction_term)},{ast.nat(value['returned'])},"
        f"{ast.sequence(value['captures'], constant_term)}⟩"
    )


def stage_term(value):
    return (
        f"⟨{ast.nat(value['ordinal'])},{ast.sequence(value['inputs'], ast.nat)},"
        f"{ast.nat(value['destination'])},{ast.nat(value['result'])},{ast.sequence(value['shape'], ast.nat)},"
        f"{body_term(value['body'])}⟩"
    )


def item_term(value):
    kind = value["kind"]
    if kind == "empty":
        return f"(.empty {ast.nat(value['result'])} {type_term(value['type'])})"
    if kind == "constant":
        return f"(.constant {constant_term(value['value'])})"
    if kind == "generic":
        return f"(.generic ⟨{stage_term(value['stage'])},{ast.string(value['key'])}⟩)"
    raise ValueError("unknown source model item")


def program_term(value):
    arguments = ast.sequence(value["arguments"], lambda pair: f"({ast.nat(pair[0])},{type_term(pair[1])})")
    return (
        f"⟨{ast.string(value['name'])},{arguments},{ast.sequence(value['items'], item_term)},"
        f"{ast.sequence(value['returned'], ast.nat)},{ast.sequence(value['results'], type_term)}⟩"
    )


def graph(value, context):
    nodes = []

    def render(shape, width):
        return "tensor<" + "".join(str(d) + "x" for d in shape) + "i" + str(width) + ">"

    for item in value["items"]:
        if item["kind"] != "generic":
            continue
        stage = item["stage"]
        nodes.append(
            dict(
                ordinal=stage["ordinal"],
                semanticKey=item["key"],
                operands=stage["inputs"],
                results=[stage["result"]],
                operandTypes=[render(stage["shape"], pair[1]) for pair in stage["body"]["inputs"]],
                resultTypes=[render(stage["shape"], stage["body"]["width"])],
            )
        )
    return dict(
        nodes=nodes,
        functionArguments=[pair[0] for pair in value["arguments"]],
        observations=value["returned"],
        context=[context[i : i + 64] for i in range(0, len(context), 64)],
    )
