"""Independent exact IEEE source-literal declarations, with no semantic grant."""

import math

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.parser import Parser

_FORMATS = {
    "f16": (2, 0x7C00, 0x03FF),
    "bf16": (2, 0x7F80, 0x007F),
    "f32": (4, 0x7F800000, 0x007FFFFF),
    "f64": (8, 0x7FF0000000000000, 0x000FFFFFFFFFFFFF),
}


def checked_literal_words(records, reports):
    source = {record["id"]: record for record in records if "floating_literal_words" in record}
    if not isinstance(reports, list):
        raise ValueError("floating literal reports require their complete source inventory")
    fields = {"dtype", "type", "word_bits", "encoding", "words", "nan_count", "shape"}
    seen = set()
    for report in reports:
        if (
            not isinstance(report, dict)
            or set(report) != fields | {"kind", "location", "source_ids"}
            or report.get("kind") != "source-floating-literal-words"
            or not isinstance(report.get("source_ids"), list)
            or any(not isinstance(value, str) for value in report["source_ids"])
        ):
            raise ValueError("floating literal reports differ from their exact word schema")
        info = {key: report[key] for key in fields}
        dtype = info.get("dtype")
        if (
            not isinstance(dtype, str)
            or dtype not in _FORMATS
            or type(info.get("word_bits")) is not int
            or info["word_bits"] != _FORMATS[dtype][0] * 8
            or info.get("encoding") != "little-endian-IEEE-words"
            or not isinstance(info.get("type"), str)
            or not isinstance(info.get("words"), str)
            or not info["words"]
            or len(info["words"]) % (_FORMATS[dtype][0] * 2)
            or any(char not in "0123456789abcdef" for char in info["words"])
        ):
            raise ValueError("floating literal reports have invalid typed storage words")
        try:
            type_ = Parser(Context(), info["type"]).parse_type()
        except Exception as error:
            raise ValueError("floating literal reports have an invalid source type") from error
        shaped = isinstance(type_, (builtin.TensorType, builtin.VectorType))
        element = type_.element_type if shaped else type_
        shape = list(type_.get_shape()) if shaped else None
        data = bytes.fromhex(info["words"])
        size, exponent, fraction = _FORMATS[dtype]
        count = len(data) // size
        nans = sum(
            word & exponent == exponent and word & fraction != 0
            for word in (int.from_bytes(data[index : index + size], "little") for index in range(0, len(data), size))
        )
        if (
            not isinstance(element, builtin.AnyFloat)
            or element.name != dtype
            or str(type_) != info["type"]
            or info.get("shape") != shape
            or shape is not None
            and any(type(value) is not int or value < 0 for value in info["shape"])
            or count not in (1, math.prod(shape) if shaped else 1)
            or type(info.get("nan_count")) is not int
            or info["nan_count"] != nans
            or nans < 1
        ):
            raise ValueError("floating literal reports differ from their original typed shape or NaN words")
        candidates = set(report["source_ids"]) & set(source)
        if len(candidates) != 1 or seen & candidates:
            raise ValueError("floating literal reports must account for each source constant once")
        record = source[next(iter(candidates))]
        spelling = (
            "0x" + data[::-1].hex().upper() + " : " + dtype
            if not shaped
            else 'dense<"0x' + data.hex().upper() + '"> : ' + info["type"]
        )
        if (
            record.get("opcode") != "arith.constant"
            or record.get("output_types") != [info["type"]]
            or record.get("floating_literal_words") != info
            or record.get("properties", {}).get("value") != spelling
        ):
            raise ValueError("floating literal reports differ from the complete source constant")
        seen.update(candidates)
    if seen != set(source):
        raise ValueError("floating literal reports do not cover every source word constant")


def checked_trace_literal_words(provenance):
    reports = [
        report
        for report in provenance.get("transformations", [])
        if report.get("kind") == "source-floating-literal-words"
    ]
    checked_literal_words(provenance.get("source_operations", []), reports)
