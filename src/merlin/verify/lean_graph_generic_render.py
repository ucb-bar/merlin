"""SOURCE ONLY: existing serializer with finite block4096/depth32 resource policy.
No compiler eligibility or altered term interpretation.
"""


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
    if type(records) is not list or not records or len(records) > 4096 or depth > 32:
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
