"""Lossless typed view/operation projection helpers, trusted Python."""

from .lean_graph_input_admission import ModelRefusal
from .lean_graph_loops_segment_projection import _key, _metadata, _types


def kind(record):
    tag = record["kind"]
    if tag == "index":
        return dict(tag="index")
    if tag == "scalar":
        if record["spelling"] not in ("i1", "i8", "i16", "i32", "i64"):
            raise ModelRefusal("only declared signless scalar widths are admitted in this graph mode")
        return dict(tag="scalar", type=record["spelling"])
    if tag != "view" or not record["static"] or not record["contiguous"] or not record["default_space"]:
        raise ModelRefusal("whole boundary requires static ranked identity-layout default-space views")
    if not record["shape"] or record["element"] not in ("i8", "i16", "i32", "i64"):
        raise ModelRefusal("ranked integer view domain required")
    return dict(tag="view", type={name: record[name] for name in ("shape", "offset", "strides", "element", "space")})


def binding(identity, types, known=None):
    return dict(name=identity, kind=kind(types[str(identity)]), known=known)


def operation_key(op):
    return _key(dict(_metadata(op), operand_arity=len(op["operands"]), result_types=_types(op["results"])))


def standard_yield(op, name):
    if op["name"] != name or any(op[field] for field in ("attributes", "properties", "results", "regions")):
        raise ModelRefusal("exact standard function return or region yield required")


def index_constant(op, types):
    if (
        op["name"] != "arith.constant"
        or op["attributes"]
        or op["operands"]
        or op["regions"]
        or set(dict(op["properties"])) != {"value"}
        or len(op["results"]) != 1
        or kind(types[str(op["results"][0][0])]) != dict(tag="index")
        or type(op["index_constant"]) is not int
        or op["index_constant"] < 0
    ):
        raise ModelRefusal("explicit fitting nonnegative index constant required")
    return op["index_constant"]
