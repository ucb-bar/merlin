"""Prove retained argument table layout against the newly selected C header.

Old recipes need not have recorded included headers. This check does not invent
one: it compiles designated semantic fields from the generated table and compares
their complete representation with the actual old object symbol. The descriptor
runtime is separately rebuilt against this same header, never an unknown old ABI.
"""

from __future__ import annotations

import hashlib
import json

from . import chain_image_layout as layout
from . import chain_image_sections as sections
from .chain_image_objects import pin, reopen


def _fields(text):
    fields, start, depth = [], 0, 0
    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "," and depth == 0:
            fields.append(text[start:index].strip())
            start = index + 1
        if depth < 0:
            raise ValueError("unbalanced generated argument table")
    if depth:
        raise ValueError("unbalanced generated argument table")
    fields.append(text[start:].strip())
    return fields


def _integer(text):
    value = text.strip().removesuffix("L")
    if not value.isascii() or not value.isdecimal() or int(value) >= 1 << 63:
        raise ValueError("generated argument table contains an unsupported expression")
    return int(value)


def designated(header, n_args):
    text = header.read_text().splitlines()
    starts = [i for i, line in enumerate(text) if line.startswith("static const merlin_arg_t MERLIN_ARGS[")]
    if len(starts) != 1:
        raise ValueError("generated argument table has no unique supported declaration")
    rows, closed = [], False
    for line in text[starts[0] + 1 :]:
        if line.strip() == "};":
            closed = True
            break
        value = line.strip()
        if not value.startswith("{") or not value.endswith("},"):
            raise ValueError("generated argument table uses unsupported initializer syntax")
        fields = _fields(value[1:-2])
        if len(fields) != 5 or fields[0] not in {"MERLIN_INPUT", "MERLIN_OUTPUT", "MERLIN_WEIGHT"}:
            raise ValueError("generated argument fields do not have the original semantic ABI")
        offset, rank, width = _integer(fields[1]), _integer(fields[2]), _integer(fields[4])
        if not fields[3].startswith("{") or not fields[3].endswith("}"):
            raise ValueError("generated argument dimensions are not literal")
        dims = [_integer(x) for x in fields[3][1:-1].split(",") if x.strip()]
        valid_dims = dims == [0] if rank == 0 else len(dims) == rank and all(d > 0 for d in dims)
        if rank > 8 or not valid_dims or width not in (1, 2, 4, 8):
            raise ValueError("generated argument table has an unsupported shape/storage ABI")
        rows.append(
            "{"
            + f".kind={fields[0]},.offset={offset}L,.rank={rank},"
            + f".dims={{{','.join(map(str, dims))}}},.elem_size={width}"
            + "}"
        )
    if not closed or len(rows) != n_args:
        raise ValueError("generated argument table count differs from its compiled extent")
    return "static const merlin_arg_t MERLIN_ARGS[] __attribute__((used)) = {\n" + ",\n".join(rows) + "\n};\n"


def verify(program, header, *, recipe, runner, directory):
    original = program["production"]
    generated = original["root"] / "cgen/model_gen.h"
    source, product = directory / f"abi{program['index']}.c", directory / f"abi{program['index']}.o"
    source.write_text(
        f"#include {json.dumps(str(header))}\n"
        + '_Static_assert(sizeof(void*) == 8 && sizeof(long) == 8 && sizeof(int) == 4, "selected LP64 ABI");\n'
        + designated(generated, program["n_args"])
    )
    recipe.run(
        layout.compile_argv(layout.command(original, "model_main.o"), source, product),
        runner=runner,
        inputs=(source, generated, header),
        output=product,
    )
    actual = reopen(original["by_name"]["model_main.o"])
    old, new = sections.symbol_bytes(actual, "MERLIN_ARGS"), sections.symbol_bytes(product, "MERLIN_ARGS")
    if old != new:
        raise ValueError("retained argument table differs from the selected header member layout")
    return {
        "status": "observed_argument_layout_equal",
        "header": pin(header),
        "generated_table": pin(generated),
        "old_object": pin(actual),
        "probe_object": pin(product),
        "table_bytes": len(old),
        "table_sha256": hashlib.sha256(old).hexdigest(),
        "scope": "Observed selected table layout; no claim that an unrecorded historical header was recovered.",
    }
