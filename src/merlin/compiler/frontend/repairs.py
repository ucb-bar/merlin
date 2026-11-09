"""Repair only the admitted legacy MLIR text syntax before structural parsing.

Fixed literal scanners preserve unknown text for the ordinary parser to reject.
Keep this compatibility route until whole-model structural replacements qualify.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

_IDENT_CHARS = frozenset(string.ascii_letters + string.digits + "_")

_DEQUANT_HEAD = ' = "quant_ext.dequantize_per_channel"('

_AXIS_HEAD = ") <{axis = "

_AXIS_TAIL = " : i64"

_PROPS_CLOSE = "}>"

_TYPES_SEP = " : ("

_RESULT_SEP = ") -> "


def _skip_space(text: str, at: int) -> int:
    while at < len(text) and text[at].isspace():
        at += 1
    return at


def _split_operand_types(line: str, at: int) -> tuple[str, str, str, str] | None:
    """Parse ``wty, sty, zty) -> outty`` starting at ``at``; ``outty`` runs to end of line.

    The first two operand types are delimited by ``, `` and the third by ``)`` — exact for
    the ranked tensor types this op carries, none of which contains a comma or a paren.
    ``None`` when the tail is not that shape, so the caller can try the next candidate.
    """
    fields = []
    for delimiter in (", ", ", "):
        end = line.find(",", at)
        if end <= at or not line.startswith(delimiter, end):
            return None
        fields.append(line[at:end])
        at = end + len(delimiter)
    end = line.find(")", at)
    if end <= at or not line.startswith(_RESULT_SEP, end):
        return None
    fields.append(line[at:end])
    outty = line[end + len(_RESULT_SEP) :]
    return (*fields, outty) if outty else None


@dataclass
class _Dequant:
    """One ``quant_ext.dequantize_per_channel`` line, split into its fields.

    The real line (gemma2 2b int8 recapture, ``out/artifacts/recaptures/*/model.mlir``)::

        %1121 = "quant_ext.dequantize_per_channel"(%1115, %2, %1120) <{axis = 1 : i64,
          input_dtype = "i8"}> {prov.op = "dequantize", …} :
          (tensor<2304x2048xi8>, tensor<2048xf32>, tensor<2048xi32>) -> tensor<2304x2048xf32>

    (wrapped here; it is ONE line in the artifact — which is exactly why the rewrite can be
    done line by line without a parser).
    """

    indent: str
    result: str
    operands: list[str]  # weights, scales, zero points
    axis: int
    weight_type: str
    scale_type: str
    zero_type: str
    result_type: str


def _parse_dequant(line: str) -> _Dequant | None:
    """Split a dequantize line into :class:`_Dequant`, or ``None`` if it is not one."""
    body = line.lstrip()
    indent = line[: len(line) - len(body)]
    head = body.find(_DEQUANT_HEAD)
    if head < 1:
        return None
    result = body[:head]
    if len(result) < 2 or not result.startswith("%") or any(c.isspace() for c in result):
        return None
    args_at = head + len(_DEQUANT_HEAD)
    axis_at = body.find(_AXIS_HEAD, args_at)
    if axis_at < 0:
        return None
    operands = body[args_at:axis_at].split(", ")
    if len(operands) != 3 or not all(
        o.startswith("%") and len(o) > 1 and not any(c.isspace() for c in o) for o in operands
    ):
        return None
    digits_at = axis_at + len(_AXIS_HEAD)
    digits_end = digits_at
    while digits_end < len(body) and body[digits_end].isdigit():
        digits_end += 1
    if digits_end == digits_at or not body.startswith(_AXIS_TAIL, digits_end):
        return None
    props_end = body.find("}", digits_end + len(_AXIS_TAIL))
    if props_end < 0 or not body.startswith(_PROPS_CLOSE, props_end):
        return None
    # The type signature is the LAST ` : (` on the line that parses as one — anything before it
    # is the (discarded) attribute dictionary, which may itself contain a colon.
    probe = len(body)
    while True:
        types_at = body.rfind(_TYPES_SEP, props_end, probe)
        if types_at < 0:
            return None
        probe = types_at
        parsed = _split_operand_types(body, types_at + len(_TYPES_SEP))
        if parsed is not None:
            return _Dequant(indent, result, operands, int(body[digits_at:digits_end]), *parsed)


def _dequant_to_generic(op: _Dequant) -> str:
    ind, res = op.indent, op.result
    init = f"%dq_init_{res[1:]}"  # %123 -> %dq_init_123 (suffixing %123 is invalid)
    wty, sty, zty, outty = (t.strip() for t in (op.weight_type, op.scale_type, op.zero_type, op.result_type))
    axis = op.axis
    shape = outty[len("tensor<") : -1].split("x")
    elem = shape[-1]
    rank = len(shape) - 1
    dims = ", ".join(f"d{i}" for i in range(rank))
    ident = f"affine_map<({dims}) -> ({dims})>"
    proj = f"affine_map<({dims}) -> (d{axis})>"
    iters = ", ".join(['"parallel"'] * rank)
    welem = wty[len("tensor<") : -1].split("x")[-1]
    zelem = zty[len("tensor<") : -1].split("x")[-1]
    selem = sty[len("tensor<") : -1].split("x")[-1]
    if (welem, zelem, selem, elem) == ("i8", "i32", "f32", "f32"):
        arithmetic = (
            f"{ind}  %dq_wide = arith.extsi %w_el : i8 to i32\n"
            f"{ind}  %dq_delta = arith.subi %dq_wide, %z_el : i32\n"
            f"{ind}  %dq_df = arith.sitofp %dq_delta : i32 to f32\n"
        )
    else:
        arithmetic = (
            f"{ind}  %dq_wf = arith.sitofp %w_el : {welem} to {elem}\n"
            f"{ind}  %dq_zf = arith.sitofp %z_el : {zelem} to {elem}\n"
            f"{ind}  %dq_df = arith.subf %dq_wf, %dq_zf : {elem}\n"
        )
    w, s, z = op.operands
    return (
        f"{ind}{init} = tensor.empty() : {outty}\n"
        f"{ind}{res} = linalg.generic {{indexing_maps = [{ident}, {proj}, {proj}, "
        f"{ident}], iterator_types = [{iters}]}} "
        f"ins({w}, {s}, {z} : {wty}, {sty}, {zty}) "
        f"outs({init} : {outty}) {{\n"
        f"{ind}^bb0(%w_el: {welem}, %s_el: {elem}, %z_el: {zelem}, %o_el: {elem}):\n"
        f"{arithmetic}"
        f"{ind}  %dq_r = arith.mulf %dq_df, %s_el : {elem}\n"
        f"{ind}  linalg.yield %dq_r : {elem}\n"
        f"{ind}}} -> {outty}"
    )


_FUNC_HEAD = "func.func @"


def _attach_c_interface(text: str) -> tuple[str, int]:
    """Attach ``attributes {llvm.emit_c_interface}`` to the FIRST ``func.func`` in ``text``.

    Matches the signature by structure: ``func.func @name(`` … ``)`` (its first ``)``, which
    is the argument list's, since a ranked tensor type carries no parenthesis), an optional
    ``-> <results>``, then the body ``{``. Returns (text, number of funcs annotated).
    """
    at = 0
    while True:
        start = text.find(_FUNC_HEAD, at)
        if start < 0:
            if '"func.func"' in text:
                # Generic functions have properties rather than a textual
                # signature. Attach the public definition's interface through
                # typed IR; private/external callbacks keep their original ABI.
                from io import StringIO

                from xdsl.dialects.builtin import UnitAttr
                from xdsl.dialects.func import FuncOp
                from xdsl.printer import Printer

                from merlin.frontends.linalg_mlir import parse_mlir_text

                module = parse_mlir_text(text)
                for fn in module.body.block.ops:
                    if (
                        not isinstance(fn, FuncOp)
                        or not fn.body.blocks
                        or (fn.sym_visibility is not None and fn.sym_visibility.data != "public")
                    ):
                        continue
                    if "llvm.emit_c_interface" in fn.attributes:
                        return text, 0
                    fn.attributes["llvm.emit_c_interface"] = UnitAttr()
                    module.verify()
                    stream = StringIO()
                    Printer(stream=stream, print_generic_format=True).print_op(module)
                    return stream.getvalue() + "\n", 1
            return text, 0
        at = start + 1
        name_at = start + len(_FUNC_HEAD)
        name_end = name_at
        while name_end < len(text) and text[name_end] in _IDENT_CHARS:
            name_end += 1
        if name_end == name_at or not text.startswith("(", name_end):
            continue
        args_close = text.find(")", name_end)
        brace = text.find("{", name_end)
        if args_close < 0 or brace < 0 or brace < args_close:
            continue  # a `{` inside the arg list: not a signature we know
        # A func that STATES its attributes (`... attributes {...}`) is left as written: the `{` found
        # above is its attribute dictionary, not its body. A declaration (no body) is skipped; a
        # definition that already asks for the C interface is the one being annotated.
        declared = text.find(" attributes {", args_close, brace + 1)
        if declared >= 0:
            header_end = text.find("}", declared)
            if header_end < 0:
                continue
            after_header = _skip_space(text, header_end + 1)
            if not text.startswith("{", after_header):
                continue  # a declaration: there is no body to call
            if "llvm.emit_c_interface" in text[declared:header_end]:
                return text, 1
            continue
        after_args = _skip_space(text, args_close + 1)
        results = ""
        if text.startswith("->", after_args):
            body_at = text.find("{", after_args)
            if body_at < 0:
                continue
            trimmed = body_at
            while trimmed > after_args and text[trimmed - 1].isspace():
                trimmed -= 1
            results = text[after_args:trimmed]
        else:
            body_at = after_args
            if not text.startswith("{", body_at):
                continue
        signature = text[start : args_close + 1]
        return (f"{text[:start]}{signature} {results} attributes {{llvm.emit_c_interface}} {{{text[body_at + 1 :]}"), 1


_EXTRACT_HEAD = '"tensor.extract_slice"('

_INSERT_HEAD = '"tensor.insert_slice"('

_PROPS_OPEN = ") <{"

_SIZES_FIELD = "static_sizes = array<i64: "

_STRIDES_FIELD = "static_strides = array<i64: "

_OFFSETS_FIELD = "static_offsets = array<i64: "

_OPERAND_TYPES = ": (tensor<"


def _array_field(props: str, field: str, *, last: bool) -> tuple[int, int] | None:
    """(value start, value end) of ``field = array<i64: …>`` inside a properties dict."""
    at = props.rfind(field) if last else props.find(field)
    if at < 0:
        return None
    value_at = at + len(field)
    value_end = props.find(">", value_at)
    return None if value_end <= value_at else (value_at, value_end)


def _slice_op_span(line: str, head_at: int, head: str, n_types: int):
    """Split a generic-form slice op into (args, props, attrs, types, end-of-op).

    ``head_at`` indexes the op's ``"tensor.<x>_slice"(``. ``n_types`` is how many operand
    types its signature carries (1 for extract_slice, 2 for insert_slice). ``None`` when the
    text at ``head_at`` is not that op spelled the way model2MLIR spells it.
    """
    args_at = head_at + len(head)
    args_end = line.find(")", args_at)
    if args_end <= args_at or not line.startswith(_PROPS_OPEN, args_end):
        return None
    props_at = args_end + len(_PROPS_OPEN)
    props_end = line.find("}", props_at)
    if props_end < 0 or not line.startswith(_PROPS_CLOSE, props_end):
        return None
    attrs_at = props_end + len(_PROPS_CLOSE)
    colon = line.find(":", attrs_at)
    if colon < 0 or not line.startswith(_OPERAND_TYPES, colon):
        return None
    types: list[str] = []
    at = colon + len(_OPERAND_TYPES) - len("tensor<")
    for index in range(n_types):
        if not line.startswith("tensor<", at):
            return None
        dims_at = at + len("tensor<")
        dims_end = line.find(">", dims_at)
        closer = ">)" if index == n_types - 1 else ">, "
        if dims_end <= dims_at or not line.startswith(closer, dims_end):
            return None
        types.append(line[dims_at:dims_end])
        at = dims_end + len(closer)
    return (line[args_at:args_end], line[props_at:props_end], line[attrs_at:colon], types, at)


def _fix_extract_slices(line: str) -> str:
    """Left-pad rank-reduced ``tensor.extract_slice`` static_sizes to the SOURCE rank.

    model2MLIR emits the sizes with only the result dims (real op from
    ``out/artifacts/recaptures/…/model.mlir``: ``static_sizes = array<i64: 32>`` against a
    ``tensor<1x1x32x32xf32>`` source), while upstream MLIR requires offsets/sizes/strides all
    at the source rank. Sizes already at (or above) the source rank are left alone.
    """
    out: list[str] = []
    i = 0
    while True:
        at = line.find(_EXTRACT_HEAD, i)
        if at < 0:
            out.append(line[i:])
            return "".join(out)
        span = _slice_op_span(line, at, _EXTRACT_HEAD, 1)
        if span is None:
            out.append(line[i : at + 1])
            i = at + 1
            continue
        args, props, attrs, (src,), end = span
        field = _array_field(props, _SIZES_FIELD, last=True)
        if field is None:
            out.append(line[i:end])
            i = end
            continue
        raw_sizes = props[field[0] : field[1]]
        src_dims = src.split("x")[:-1]
        sizes = [s.strip() for s in raw_sizes.split(",")]
        if len(sizes) >= len(src_dims):
            out.append(line[i:end])
            i = end
            continue
        padded = ["1"] * (len(src_dims) - len(sizes)) + sizes
        fixed = props.replace(f"{_SIZES_FIELD}{raw_sizes}>", _SIZES_FIELD + ", ".join(padded) + ">")
        out.append(line[i:at])
        out.append(f"{_EXTRACT_HEAD}{args}{_PROPS_OPEN}{fixed}{_PROPS_CLOSE}{attrs}{_OPERAND_TYPES}{src}>)")
        i = end


def _fix_insert_slices(line: str) -> str:
    """Reset ``tensor.insert_slice`` strides that overrun the destination back to 1.

    Pre-fixes model2MLIR artifacts that predate the slice_scatter step fix (m2m's
    ``ir/decompositions.py`` read ``step`` from the ``end`` argument slot, so insert_slice got
    ``strides[dim] = end``). ``step`` is almost always 1; only a stride the slice cannot FIT under
    is reset. New captures do not hit this, but the recaptures already on disk do, so the repair
    stays.

    "Fits" is ``offset + (size - 1) * stride < extent`` -- the LAST written index, not one stride
    past it. The earlier ``size * stride > extent`` counted the step past the final element as a
    written element and so rejected a scatter that is exactly full: a ``ConvTranspose2d(stride=2)``
    upsample writes ``size=16`` elements at 0, 2, ..., 30 of a 31-wide destination, and ``16 * 2 =
    32 > 31`` reset it to 1, turning the scatter into a dense corner copy. Silently: the module
    stays verifier-clean, so it surfaced only as a wrong number (measured on deepjscc, whose two
    transposed convolutions are the only non-unit-stride inserts in any tracked recapture --
    whole-model ``fp32_cos 0.885366`` against the capture's own golden, ``1.000000`` with this
    predicate). ``merlin/tests/ir/test_insert_slice_strides.py`` pins both directions.
    """
    out: list[str] = []
    i = 0
    while True:
        at = line.find(_INSERT_HEAD, i)
        if at < 0:
            out.append(line[i:])
            return "".join(out)
        span = _slice_op_span(line, at, _INSERT_HEAD, 2)
        if span is None:
            out.append(line[i : at + 1])
            i = at + 1
            continue
        args, props, attrs, (src, dst), end = span
        sizes_field = _array_field(props, _SIZES_FIELD, last=True)
        strides_field = (
            None if sizes_field is None else _array_field(props[sizes_field[1] :], _STRIDES_FIELD, last=False)
        )
        offsets_field = _array_field(props, _OFFSETS_FIELD, last=False)
        if sizes_field is None or strides_field is None:
            out.append(line[i:end])
            i = end
            continue
        raw_strides = props[sizes_field[1] + strides_field[0] : sizes_field[1] + strides_field[1]]
        sizes = [int(s) for s in props[sizes_field[0] : sizes_field[1]].split(",")]
        strides = [int(s) for s in raw_strides.split(",")]
        dst_dims = [int(d) for d in dst.split("x")[:-1]]
        # The offset is part of where the last element lands; an absent field means all-zero.
        offsets = (
            [int(s) for s in props[offsets_field[0] : offsets_field[1]].split(",")]
            if offsets_field is not None
            else [0] * len(sizes)
        )
        repaired = [
            1 if off + (sz - 1) * st >= extent else st for off, sz, st, extent in zip(offsets, sizes, strides, dst_dims)
        ]
        if repaired == strides:
            out.append(line[i:end])
            i = end
            continue
        fixed = props.replace(
            f"{_STRIDES_FIELD}{raw_strides}>", _STRIDES_FIELD + ", ".join(str(s) for s in repaired) + ">", 1
        )
        out.append(line[i:at])
        out.append(
            f"{_INSERT_HEAD}{args}{_PROPS_OPEN}{fixed}{_PROPS_CLOSE}{attrs}{_OPERAND_TYPES}{src}>, tensor<{dst}>)"
        )
        i = end


def preprocess_text_textual(mlir_text: str) -> tuple[str, dict]:
    """Whole-model variant of :func:`preprocess_text` (pure text; no xDSL roundtrip)."""
    out_lines = []
    n = 0
    for line in mlir_text.splitlines():
        op = _parse_dequant(line)
        if op is not None:
            out_lines.append(_dequant_to_generic(op))
            n += 1
        else:
            out_lines.append(_fix_insert_slices(_fix_extract_slices(line)))
    text, n_funcs = _attach_c_interface("\n".join(out_lines))
    return text, {"dequantize_lowered": n, "c_interface_funcs": n_funcs}
