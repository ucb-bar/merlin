"""SDK-free literal serializer for the unchanged GraphState.Scalar checker.

The pure serializer functions below are extracted unchanged from the owned
lean_graph_state_scalar.py leaf. Separating them avoids importing its historical
native runner merely to emit a candidate query. No production transformation or
Lean semantics is substituted. New source identity; no historical qualification.
"""
from __future__ import annotations


def integer(value, *, natural=False):
    if type(value) is not int or natural and value < 0:
        raise ValueError("exact integer graph field required; bool/float refused")
    return str(value) if natural else "(" + str(value) + ")"


def closed(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("complete closed typed graph record required")


def type_code(value):
    if type(value) is not dict or len(value) != 1:
        raise ValueError("exact one typed value constructor required")
    if set(value) == {"index"} and value["index"] is True:
        return ".index"
    if set(value) == {"bits"} and type(value["bits"]) is int and value["bits"] > 0:
        return "(.bits " + str(value["bits"]) + ")"
    if set(value) == {"opaque"}:
        return "(.opaque " + integer(value["opaque"], natural=True) + ")"
    raise ValueError("unsupported exact scalar value type")


def types(values):
    if type(values) is not list:
        raise ValueError("ordered complete type list required")
    return "[" + ",".join(type_code(value) for value in values) + "]"


def indices(values):
    if type(values) is not list:
        raise ValueError("complete ordered SSA operand list required")
    return "[" + ",".join(integer(value, natural=True) for value in values) + "]"


def primitive(value):
    if type(value) is not dict:
        raise ValueError("closed pure primitive record required")
    kind = value["kind"]
    if kind == "constant":
        closed(value, {"kind", "width", "value"})
        return "(.constant " + integer(value["width"], natural=True) + " " + integer(value["value"]) + ")"
    if kind in ("add", "sub", "mul"):
        closed(value, {"kind", "width"})
        return "(." + kind + " " + integer(value["width"], natural=True) + ")"
    if kind in ("sext", "zext", "trunc"):
        closed(value, {"kind", "source", "target"})
        return (
            "(."
            + kind
            + " "
            + integer(value["source"], natural=True)
            + " "
            + integer(value["target"], natural=True)
            + ")"
        )
    raise ValueError("pure primitive lies outside exact unflagged fixed-width whitelist")


def program(records, *, depth=0):
    if type(records) is not list or not records or len(records) > 256 or depth > 16:
        raise ValueError("complete bounded nonempty supported block required")
    closed(records[-1], {"ret"})
    result = "(.ret " + indices(records[-1]["ret"]) + ")"
    for record in reversed(records[:-1]):
        if "pure" in record:
            closed(record, {"pure", "arguments"})
            result = "(.pure " + primitive(record["pure"]) + " " + indices(record["arguments"]) + " " + result + ")"
        elif "effect" in record:
            closed(record, {"effect", "arguments", "outputs"})
            result = (
                "(.opaque "
                + integer(record["effect"], natural=True)
                + " "
                + indices(record["arguments"])
                + " "
                + types(record["outputs"])
                + " "
                + result
                + ")"
            )
        elif "loop" in record:
            closed(record, {"loop"})
            loop = record["loop"]
            closed(loop, {"bounds", "initial", "carries", "body"})
            if type(loop["bounds"]) is not list or len(loop["bounds"]) != 3:
                raise ValueError("complete exact lower/upper/stride operands required")
            bounds = " ".join(integer(value, natural=True) for value in loop["bounds"])
            result = (
                "(.loop "
                + bounds
                + " "
                + indices(loop["initial"])
                + " "
                + types(loop["carries"])
                + " "
                + program(loop["body"], depth=depth + 1)
                + " "
                + result
                + ")"
            )
        else:
            raise ValueError("unsupported/missing complete statement")
    return result


def budget(records, count, work, captured_sizes=None):
    """Conservative expression-expansion bound; never scales with loop trips."""
    environment = [1] * count if captured_sizes is None else list(captured_sizes)
    for record in records:
        if "loop" in record:
            loop = record["loop"]
            budget(
                loop["body"],
                len(environment) + len(loop["carries"]) + 1,
                work,
                [1] * (len(loop["carries"]) + 1) + environment,
            )
            environment = [1] * len(loop["carries"]) + environment
            continue
        args = record.get("arguments", record.get("ret", []))
        indices(args)
        if any(index >= len(environment) for index in args):
            # Let the proved checker demonstrate valid-data dominance rejection.
            return
        size = 1 + sum(environment[index] for index in args)
        work[0] += size
        if size > 4096 or work[0] > 65536:
            raise ValueError("supported scalar symbolic expression budget exceeded")
        if "pure" in record:
            environment.insert(0, size)
        elif "effect" in record:
            environment = [1] * len(record["outputs"]) + environment


def emit_query(projection):
    registry = []
    for row in projection["registry"]:
        registry.append("{inputs := " + types(row["inputs"]) + ",outputs := " + types(row["outputs"]) + "}")
    text = (
        "import Merlin.GraphStateScalar\nset_option maxRecDepth 4096\nset_option maxHeartbeats 2000000\n"
        "namespace Merlin.GraphStateScalarInstance\nopen Merlin.GraphState.Scalar\n"
        "def registry : List Signature := [" + ",".join(registry) + "]\n"
    )
    roots = [
        "Merlin.GraphState.Scalar.check_sound",
        "Merlin.GraphState.Scalar.check_typed",
        "Merlin.GraphState.Scalar.normalize_sound",
    ]
    defined = 0
    for index, function in enumerate(projection["functions"]):
        if function["declaration"]:
            continue
        defined += 1
        for key in ("before", "after"):
            budget(function[key], len(function["inputs"]), [0])
        namespace = "F" + str(index)
        text += (
            "namespace " + namespace + "\n"
            "def inputs : List Ty := " + types(function["inputs"]) + "\n"
            "def outputs : List Ty := " + types(function["outputs"]) + "\n"
            "def before : Program := " + program(function["before"]) + "\n"
            "def after : Program := " + program(function["after"]) + "\n"
            "theorem accepted : check registry inputs outputs before after = true := by decide\n"
            "theorem actual : Preserves registry inputs.length before after := check_sound accepted\n"
            "theorem typed : Typed registry inputs outputs before ∧ Typed registry inputs outputs after :=\n"
            "  check_typed accepted\n"
            "theorem initialized {Value World Error : Type} (model : Model Value World Error registry)\n"
            "    (values : List Value) (world : World) (arity : values.length = inputs.length) :\n"
            "    Shaped outputs.length (evalProgram model before values world) ∧\n"
            "    Shaped outputs.length (evalProgram model after values world) :=\n"
            "  check_initialized accepted model values world arity\n"
            "theorem operands_initialized {Value World Error : Type} (model : Model Value World Error registry)\n"
            "    (values : List Value) (world : World) (arity : values.length = inputs.length) :\n"
            "    Initialized model before values world ∧ Initialized model after values world :=\n"
            "  check_operands_initialized accepted model values world arity\n"
            "end " + namespace + "\n"
        )
        roots += [
            "Merlin.GraphStateScalarInstance." + namespace + "." + name
            for name in ("accepted", "actual", "typed", "initialized", "operands_initialized")
        ]
    if not defined:
        raise ValueError("no defined complete function available for a graph proof")
    return text + "end Merlin.GraphStateScalarInstance\n", roots

