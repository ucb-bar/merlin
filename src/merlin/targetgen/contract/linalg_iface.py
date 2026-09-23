"""``linalg-on-tensors`` interface grammar: a structural reader for the *second* frozen input
format the experiment ABI hands an out-of-tree backend package.

Two input grammars ship over the same ``capsule.interface.mlir`` slot:

- ``merlin_iface`` v0.1 (see :mod:`.interface_emit`) — a tiny residency command list
  (``merlin_iface.tensor/matmul/commit``). Matmul-family only.
- ``linalg-on-tensors`` (this module) — a ``func.func @forward`` whose body is standard MLIR
  ``linalg``/``tensor``/``math`` ops, tagged with ``prov.op``/``prov.family`` provenance. This is
  what model-slice capsules carry for the op classes that have no residency mnemonic
  (softmax, rmsnorm, layernorm, gelu, rope, elementwise …) plus matmul expressed as
  ``linalg.matmul``/``linalg.generic``.

This reader is the front-door **parse** for that grammar: it returns a structural inventory of the
workload (entry signature + one record per payload op, with provenance, operand/result shapes and
dtypes, matmul extents, and — for ``linalg.generic``/``reduce`` — the inner arithmetic op names that
name the elementwise/reduction semantics). It is the exact analogue of
:func:`.interface_emit.parse_interface_mlir`: a **reader**, not a lowering. A backend package walks
this inventory and authors its own lowering to the target command buffer.

Design constraints (mirror the contract surface rules):
- Target-agnostic: the grammar is fixed by the contract, identical for every accelerator. No target
  semantics, no target-name literals.
- Structural, NOT textual: parse the real IR with xDSL (``linalg``/``tensor``/``math`` dialects) and
  read fields off the typed ops. No regex, and no textual pre-normalization: where xDSL's own custom
  syntax is wrong we fix the PARSER, never the input text (see the multi-result ``linalg.generic``
  shim below, which xDSL 0.68 prints but cannot read back).
"""

from __future__ import annotations

from typing import Any

LEVEL = "linalg-on-tensors"

# linalg named ops whose semantics are a matmul-family contraction (M,K)x(K,N)->(M,N), possibly
# batched. A contraction may ALSO arrive as a `linalg.generic` with a reduction iterator — in that
# case the record's ``prov_family == "contraction"`` still marks it, and ``reduction_dims`` is set.
_MATMUL_KINDS = ("linalg.matmul", "linalg.batch_matmul", "linalg.quantized_matmul", "linalg.matmul_transpose_b")


def make_linalg_context():
    """A permissive xDSL context that loads exactly the dialects the linalg-on-tensors grammar uses.

    ``allow_unregistered=True`` tolerates the ``prov.*`` attribute namespace the capsules carry.
    """
    from xdsl.context import Context
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.cf import Cf
    from xdsl.dialects.func import Func
    from xdsl.dialects.math import Math
    from xdsl.dialects.scf import Scf
    from xdsl.dialects.tensor import Tensor as TensorDialect

    # Teach the parser the fp8 element types (and let them satisfy arith/math's float constraints)
    # BEFORE any parse, exactly as `xdsl_dialects._common.make_context` and `frontends.linalg_mlir`
    # do. This is the THIRD context builder in the tree, and reader parity is the whole point: an
    # ordinary `arith.truncf %x : f32 to f8E4M3FN` requantize -- the bf16 -> fp8 pack a whole class of
    # narrow-datapath capsules is written around -- loads through the other two readers and fails
    # `type expected` here, so the capsule reads as malformed rather than as unregistered.
    from merlin.xdsl_dialects.fp8 import register_fp8_float_constraints, register_fp8_types

    register_fp8_types()
    register_fp8_float_constraints()

    ctx = Context(allow_unregistered=True)
    for d in (Builtin, Func, Arith, _linalg_dialect(), TensorDialect, Scf, Math, Cf):
        ctx.load_dialect(d)
    return ctx


def is_linalg_on_tensors(text: str) -> bool:
    """Cheap discriminator: does this interface MLIR carry the linalg-on-tensors provenance level?

    Lets a package's ``parse`` route between the two grammars without committing to a full parse.
    """
    return f'prov.level = "{LEVEL}"' in text


# --------------------------------------------------------------------------- structural field reads


def _shape(t) -> list[int]:
    from xdsl.dialects.builtin import TensorType

    return [int(d) for d in t.get_shape()] if isinstance(t, TensorType) else []


def _dtype(t) -> str:
    from xdsl.dialects.builtin import TensorType

    return str(t.element_type) if isinstance(t, TensorType) else str(t)


def _prov(op) -> dict[str, str]:
    from xdsl.dialects.builtin import StringAttr

    out: dict[str, str] = {}
    for table in (op.attributes, getattr(op, "properties", {}) or {}):
        for key, val in table.items():
            if key.startswith("prov.") and isinstance(val, StringAttr):
                out[key[len("prov.") :]] = val.data
    return out


def _scalar_const(value) -> float | None:
    """The compile-time scalar constant behind ``value`` when it is an ``arith.constant`` or a
    ``tensor.splat`` of one (e.g. a per-tensor scale ``x * 4.0`` splats a constant), else None.
    Lets a lowering bake the scalar into the kernel instead of treating it as a runtime operand."""
    owner = getattr(value, "owner", None)
    name = getattr(owner, "name", "")
    if name == "tensor.splat" and owner.operands:
        owner = getattr(owner.operands[0], "owner", None)
        name = getattr(owner, "name", "")
    if name == "arith.constant":
        attrs = {**owner.attributes, **(getattr(owner, "properties", {}) or {})}
        v = attrs.get("value")
        data = getattr(getattr(v, "value", None), "data", None)
        if data is not None:
            try:
                return float(data)
            except (TypeError, ValueError):
                return None
    return None


def _ins_outs(op) -> tuple[list, list]:
    """The linalg operand split (ins= data operands, outs= init/destination operands).

    Named + generic linalg ops expose ``.inputs``/``.outputs``; anything else falls back to all
    operands as inputs and none as outputs.
    """
    ins = list(getattr(op, "inputs", []) or [])
    outs = list(getattr(op, "outputs", []) or [])
    if not ins and not outs:
        ins = list(op.operands)
    return ins, outs


def _body_op_names(op) -> list[str]:
    """Inner arithmetic op names of a ``linalg.generic``/``reduce``/``map`` body (``arith.addf``,
    ``arith.maximumf``, ``math.exp`` …), in order, minus the terminator. These NAME the elementwise
    / reduction semantics that the op family alone does not (add vs mul vs sub vs div; max vs sum;
    exp/rsqrt/tanh/sin/cos)."""
    names: list[str] = []
    for region in getattr(op, "regions", []) or []:
        for block in region.blocks:
            for inner in block.ops:
                if inner.name in ("linalg.yield", "func.return"):
                    continue
                names.append(inner.name)
    return names


def _reduction_dims(op) -> list[int]:
    """The reduction iterator positions of a linalg op (empty for pure-parallel elementwise).

    Read structurally from ``iterator_types`` when present; ``linalg.reduce`` carries explicit
    ``dimensions``.
    """
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr

    attrs = {**op.attributes, **(getattr(op, "properties", {}) or {})}
    dims = attrs.get("dimensions")
    if dims is not None and hasattr(dims, "get_values"):  # DenseArrayBase (linalg.reduce dimensions)
        return [int(v) for v in dims.get_values()]
    if isinstance(dims, (list, tuple)):
        return [int(d.value.data) if isinstance(d, IntegerAttr) else int(d) for d in dims]
    if isinstance(dims, ArrayAttr):
        return [int(d.value.data) if isinstance(d, IntegerAttr) else int(d) for d in dims.data]
    its = attrs.get("iterator_types")
    out: list[int] = []
    if isinstance(its, ArrayAttr):
        for i, it in enumerate(its.data):
            s = it.data.data if hasattr(it, "data") and hasattr(it.data, "data") else str(it)
            if "reduction" in str(s):
                out.append(i)
    return out


def _matmul_extents(ins: list, outs: list) -> dict[str, int]:
    """(M,K,N) for a 2-input contraction, derived from operand shapes. Batched forms prepend batch
    dims; take the trailing 2 of each. Returns {} when the shapes are not a clean contraction."""
    if len(ins) < 2:
        return {}
    a, b = _shape(ins[0].type), _shape(ins[1].type)
    if len(a) < 2 or len(b) < 2:
        return {}
    m, k = a[-2], a[-1]
    k2, n = b[-2], b[-1]
    ext = {"m": int(m), "k": int(k), "n": int(n)}
    if len(a) >= 3 or len(b) >= 3:
        ext["batch"] = int(a[0] if len(a) >= 3 else b[0])
    if k2 != k:  # e.g. transpose_b (K,N) stored as (N,K); surface both so the lowering can adapt
        ext["k_rhs"] = int(k2)
    return ext


# --------------------------------------------------------------------------- multi-result generic

# xDSL 0.68 ships a ``linalg.generic`` custom syntax that does not ROUND-TRIP: its printer emits the
# multi-result result list parenthesized (``-> (T0, T1)``, see ``GenericOp.print``), but its parser
# reads the arrow with ``parse_attribute``, and a leading ``(`` starts an MLIR *function type*
# ``(...) -> ...`` — so the parser consumes the result list and then demands an ``->`` that is not
# there. Two consequences, both of which this shim removes:
#
#   1. every multi-result ``linalg.generic`` is unparseable — which is every arg-reduce
#      (``aten.min.dim`` / ``argmin`` / ``argmax`` lowers to a generic yielding (value, index)),
#      on every target and at every dtype;
#   2. the failure is reported at the token where the parse DESYNCHRONISED, i.e. the FOLLOWING op —
#      so the traceback blames an innocent neighbour (a ``tensor.expand_shape``, typically) and the
#      real cause is invisible.
#
# The shim is a subclass, so ``isinstance(op, GenericOp)`` and the ``HasParent(GenericOp)`` trait on
# ``linalg.index`` (which arg-reduce bodies use to materialise the index) keep holding.

_LINALG_DIALECT = None


def _multi_result_generic_op():
    """A ``linalg.generic`` whose ``parse`` accepts the parenthesized multi-result arrow.

    Everything before the arrow is parsed exactly as upstream does; only the result-type list is
    read structurally in both spellings (``-> T``, ``-> (T0, T1, ...)``, ``-> ()``).
    """
    from typing import cast

    from xdsl.dialects.builtin import ArrayAttr, StringAttr
    from xdsl.dialects.linalg.ops import GenericOp, IteratorType, IteratorTypeAttr
    from xdsl.ir import Attribute

    class MultiResultGenericOp(GenericOp):
        @classmethod
        def parse(cls, parser):
            attrs_start_pos = parser.pos
            attrs = parser.parse_optional_attr_dict()
            attrs_end_pos = parser.pos

            if "indexing_maps" not in attrs:
                parser.raise_error("Expected indexing_maps for linalg.generic", attrs_start_pos, attrs_end_pos)
            indexing_maps = attrs.pop("indexing_maps")
            assert isinstance(indexing_maps, ArrayAttr)

            if "iterator_types" not in attrs:
                parser.raise_error("Expected iterator_types for linalg.generic", attrs_start_pos, attrs_end_pos)
            parsed_iterator_types = attrs.pop("iterator_types")
            assert isinstance(parsed_iterator_types, ArrayAttr)
            parsed_iterator_types = cast(ArrayAttr[Attribute], parsed_iterator_types)
            iterator_types: list = []
            for iterator_type in parsed_iterator_types:
                if isinstance(iterator_type, IteratorTypeAttr):
                    iterator_types.append(iterator_type)
                elif isinstance(iterator_type, StringAttr):
                    iterator_types.append(IteratorTypeAttr(IteratorType(iterator_type.data)))
                else:
                    parser.raise_error(f"Unknown iterator type {iterator_type}", attrs_start_pos, attrs_end_pos)

            doc = attrs.pop("doc", None)
            library_call = attrs.pop("library_call", None)

            pos = parser.pos
            if parser.parse_optional_characters("ins"):
                parser.parse_punctuation("(")
                unresolved_ins = parser.parse_comma_separated_list(
                    parser.Delimiter.NONE, parser.parse_unresolved_operand
                )
                parser.parse_punctuation(":")
                ins_types = parser.parse_comma_separated_list(parser.Delimiter.NONE, parser.parse_type)
                parser.parse_punctuation(")")
                ins = parser.resolve_operands(unresolved_ins, ins_types, pos)
            else:
                ins = ()

            pos = parser.pos
            if parser.parse_optional_characters("outs"):
                parser.parse_punctuation("(")
                unresolved_outs = parser.parse_comma_separated_list(
                    parser.Delimiter.NONE, parser.parse_unresolved_operand
                )
                parser.parse_punctuation(":")
                outs_types = parser.parse_comma_separated_list(parser.Delimiter.NONE, parser.parse_type)
                parser.parse_punctuation(")")
                outs = parser.resolve_operands(unresolved_outs, outs_types, pos)
            else:
                outs = ()

            if parser.parse_optional_keyword("attrs"):
                parser.parse_punctuation("=")
                extra_attrs = parser.expect(parser.parse_optional_attr_dict, "expect extra attributes")
            else:
                extra_attrs = {}

            body = parser.parse_region()

            # THE FIX. Upstream reads this with ``parse_attribute``, which mis-reads the
            # parenthesized (multi-result) spelling as the operand half of a function type.
            res_types = _parse_result_arrow(parser)

            generic = cls(ins, outs, body, indexing_maps, iterator_types, res_types, doc, library_call)
            generic.attributes |= extra_attrs
            return generic

    return MultiResultGenericOp


def _parse_result_arrow(parser) -> list:
    """The trailing ``-> ...`` result-type list of a linalg op, in every spelling MLIR prints.

    ``-> T`` (one result), ``-> (T0, T1, ...)`` (what ``GenericOp.print`` emits for more than one),
    ``-> ()`` (explicitly none), and no arrow at all (none). Parenthesization is a printing choice,
    not a semantic one, so both bare and parenthesized lists are accepted at any arity.
    """
    if not parser.parse_optional_punctuation("->"):
        return []
    if parser.parse_optional_punctuation("("):
        if parser.parse_optional_punctuation(")"):
            return []
        res_types = parser.parse_comma_separated_list(parser.Delimiter.NONE, parser.parse_type)
        parser.parse_punctuation(")")
        return res_types
    return parser.parse_comma_separated_list(parser.Delimiter.NONE, parser.parse_type)


def _linalg_dialect():
    """The ``linalg`` dialect with the round-trippable ``linalg.generic`` substituted in."""
    global _LINALG_DIALECT
    if _LINALG_DIALECT is None:
        from xdsl.dialects.linalg import Linalg
        from xdsl.dialects.linalg.ops import GenericOp
        from xdsl.ir import Dialect

        fixed = _multi_result_generic_op()
        _LINALG_DIALECT = Dialect(
            Linalg.name,
            [fixed if op is GenericOp else op for op in Linalg.operations],
            list(Linalg.attributes),
        )
    return _LINALG_DIALECT


# --------------------------------------------------------------------------- fail-closed diagnostic


class LinalgParseError(ValueError):
    """A ``linalg-on-tensors`` interface this reader could not parse, NAMED AT THE OP.

    An MLIR custom parser reports the token at which it desynchronised, which for an op whose custom
    syntax it cannot read is the op AFTER the culprit. Reporting that token verbatim blames an
    innocent neighbour and sends the reader hunting the wrong construct. This error therefore names
    both: ``op`` (the op the parse actually stopped inside — the reported line, or the nearest
    preceding op when the reported line cannot own the failure) and ``reported_op``/``line``.
    """

    def __init__(self, *, op: str, reported_op: str, line: int, column: int, detail: str, window: str):
        self.op = op
        self.reported_op = reported_op
        self.line = line
        self.column = column
        self.detail = detail
        self.window = window
        blame = (
            f"{op!r}"
            if op == reported_op
            else f"{op!r} -- the parse desynchronised and stopped at the START of the following "
            f"{reported_op!r}, which is where an unreadable custom syntax SURFACES, not "
            f"where it originates"
        )
        super().__init__(
            f"cannot parse linalg-on-tensors interface at line {line}, col {column}: "
            f"unsupported or malformed op {blame} -- {detail}\n{window}"
        )


def _op_mnemonic(line: str) -> tuple[str, int]:
    """The op mnemonic of one MLIR line and the 0-based column it starts at, read structurally
    (``split``/``partition``, no regex).

    An op line is either ``%r0, %r1 = <mnemonic> ...`` (SSA result binding first) or a bare
    ``<mnemonic> ...``. Returns ``("", -1)`` for a line that owns no op (blank, block label, closing
    brace, or a continuation of a multi-line construct).
    """
    stripped = line.strip()
    if not stripped or stripped[0] in "^}])" or stripped.startswith("//"):
        return ("", -1)
    if stripped.startswith("%"):
        # `%r0, %r1 = <mnemonic> ...` — step over the SSA result binding.
        head, sep, after = line.partition(" = ")
        if not sep:
            return ("", -1)
        col = len(head) + len(sep) + (len(after) - len(after.lstrip()))
        rest = after.lstrip()
    else:
        col = len(line) - len(line.lstrip())
        rest = stripped
    parts = rest.split()
    if not parts:
        return ("", -1)
    token = parts[0].strip('{(,:"')
    # An op mnemonic is `dialect.op` or a bare builtin keyword (`return`). Anything still carrying
    # syntax after stripping is a continuation line, not an op.
    if not token or not all(c.isalnum() or c in "._" for c in token):
        return ("", -1)
    return (token, col)


def _line_depths(lines: list[str]) -> list[int]:
    """Brace nesting depth at the START of each line, counted structurally.

    Needed to walk BACK to the op that owns a desynchronised parse: the ops inside a
    ``linalg.generic`` region (``linalg.index``, ``arith.select``, ``linalg.yield`` …) sit one level
    deeper than the generic itself, so a naive line-wise walk back blames the region body instead of
    the op whose syntax actually failed. Braces inside double-quoted strings are not nesting.
    """
    depths: list[int] = []
    depth = 0
    for line in lines:
        depths.append(depth)
        in_string = False
        escaped = False
        for ch in line:
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
            elif ch == '"':
                in_string = not in_string
            elif not in_string and ch == "{":
                depth += 1
            elif not in_string and ch == "}":
                depth -= 1
    return depths


def _underlying_reason(exc) -> str:
    """The human reason from an xDSL parse failure, without its echoed source context.

    ``ParseError`` renders as a location line, the offending source line, a caret run, and THEN the
    reason -- so the first line is the least informative part and the last is the whole point
    (``Expected indexing_maps for linalg.generic``). We print our own, correctly attributed window,
    so keep the reason and drop the echo.
    """
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    reason = ""
    for ln in reversed(lines):
        if set(ln) != {"^"}:  # skip the caret run
            reason = ln
            break
    return f"{type(exc).__name__}: {reason}" if reason else type(exc).__name__


def _parse_diagnostic(text: str, exc) -> "LinalgParseError":
    """Turn an xDSL parse failure into a diagnostic that names the op, not the next line.

    The offset comes from the exception's own ``span`` (structural), never from its message text.
    """
    offset = getattr(getattr(exc, "span", None), "start", None)
    if not isinstance(offset, int):
        offset = 0
    lines = text.splitlines()
    line_no = text.count("\n", 0, offset) + 1
    line_start = text.rfind("\n", 0, offset) + 1
    column = offset - line_start + 1

    idx = min(max(line_no - 1, 0), max(len(lines) - 1, 0))
    reported_op, mnemonic_col = _op_mnemonic(lines[idx]) if lines else ("", -1)

    # WHICH op is actually to blame. Two failure shapes land here and they need opposite answers:
    #
    #  * the parser stopped INSIDE an op it was reading (a malformed field, a missing required
    #    attribute). The position is then PAST that op's mnemonic, and the reported op is the
    #    culprit.
    #  * the parser DESYNCHRONISED on an op whose custom syntax it could not read: it consumed the
    #    op, failed to consume the tail, and stopped at the very START of the NEXT op. The position
    #    is then at or before that next op's mnemonic, and the culprit is the op BEFORE it.
    #
    # Distinguishing them is the whole point: reporting the desync position verbatim is what made a
    # multi-result `linalg.generic` look like a broken `tensor.expand_shape`.
    desynced = reported_op != "" and column - 1 <= mnemonic_col
    culprit = reported_op or "<unknown>"
    if desynced or not reported_op:
        # Walk back at the SAME nesting depth: an op's own region body is deeper, and blaming a
        # `linalg.yield` inside the generic is as wrong as blaming the op after it.
        depths = _line_depths(lines)
        here = depths[idx] if idx < len(depths) else 0
        for back in range(idx - 1, -1, -1):
            prev, _ = _op_mnemonic(lines[back])
            if prev and depths[back] == here:
                culprit = prev
                break

    window = "\n".join(f"  {n + 1:>6} | {lines[n]}" for n in range(max(idx - 2, 0), min(idx + 1, len(lines))))
    return LinalgParseError(
        op=culprit or "<unknown>",
        reported_op=reported_op or "<unknown>",
        line=line_no,
        column=column,
        detail=_underlying_reason(exc),
        window=window,
    )


# --------------------------------------------------------------------------- parse


def parse_linalg_mlir(text: str, *, ctx=None) -> dict[str, Any]:
    """Parse ``linalg-on-tensors`` interface MLIR into a structural workload inventory.

    Returns::

        {
            "level": "linalg-on-tensors",
            "entry": "forward",
            "args": [{"index": 0, "shape": [16, 16], "dtype": "bf16"}, ...],  # @forward operands
            "results": [{"shape": [16, 16], "dtype": "bf16"}, ...],  # @forward results
            "ops": [  # payload ops, in order
                {
                    "id": 0,
                    "kind": "linalg.matmul",
                    "op": "matmul",
                    "family": "contraction",
                    "prov": {...},
                    "ins": [{"source": ("arg", 0), "shape": [16, 16], "dtype": "bf16"}, ...],
                    "outs": [{"source": ("op", -1) | ("init", "fill"), "shape": [...], "dtype": "..."}],
                    "results": [{"shape": [...], "dtype": "..."}],
                    "extents": {"m": 16, "k": 16, "n": 16},
                    "body_ops": [],
                    "reduction_dims": [],
                },
                ...,
            ],
        }

    ``source`` is the structural dataflow edge for each operand: ``("arg", i)`` a ``@forward`` operand,
    ``("op", j)`` the result of payload op ``j``, ``("init", kind)`` a ``tensor.empty``/``linalg.fill``
    destination, or ``("const", None)`` an ``arith.constant``. This is the DAG a backend lowers.
    An ``("op", j)`` edge additionally carries ``result_index``: WHICH of op ``j``'s results it reads.
    Multi-result ops are real (an arg-reduce ``linalg.generic`` yields value AND index together), and
    for them the op id alone does not identify the value.
    """
    from xdsl.ir import BlockArgument

    try:
        from ...common.ir_lock import IR_LOCK
    except ImportError:  # sandbox: staged flat on sys.path (no parent package) — a per-process lock is
        import threading  # semantically sufficient (each entrypoint parses single-process)

        IR_LOCK = threading.Lock()
    from xdsl.parser import Parser

    with IR_LOCK:
        try:
            module = Parser(ctx or make_linalg_context(), text).parse_module()
        except LinalgParseError:
            raise
        except Exception as exc:  # xDSL ParseError and friends: re-raise NAMED AT THE OP
            raise _parse_diagnostic(text, exc) from exc

    fns = [op for op in module.walk() if op.name == "func.func"]
    if not fns:
        raise ValueError("linalg-on-tensors interface has no func.func entry")
    fn = fns[0]
    entry = _fn_name(fn)
    block = fn.body.blocks[0]
    func_args = list(block.args)

    # The payload ops we surface are the DIRECT children of the @forward entry block (never the
    # arithmetic ops nested inside a linalg.generic/reduce region body — those are captured per op in
    # ``body_ops``, and walking them as top-level ops would double-count e.g. a softmax's inner
    # ``math.exp``). Structural init ops (tensor.empty / arith.constant / linalg.fill / tensor.splat)
    # are destinations/constants: referenced as operand ``source``s but not lowered as commands.
    _INIT = ("tensor.empty", "arith.constant", "linalg.fill", "tensor.splat")
    _SKIP = _INIT + ("func.return", "linalg.yield")
    payload = [op for op in block.ops if op.name not in _SKIP]

    # Map every SSA result value -> (payload-op id, WHICH RESULT of it) for dataflow edges. The
    # result position is load-bearing for a multi-result op: an arg-reduce generic yields (value,
    # index) from ONE op, so ``("op", j)`` alone cannot say which of the two a consumer reads — and
    # confusing an argmin's value with its index is a silent numerical wrong answer, not a crash.
    result_owner: dict[Any, tuple[int, int]] = {}
    for i, op in enumerate(payload):
        for k, res in enumerate(op.results):
            result_owner[res] = (i, k)

    def _source(value):
        if isinstance(value, BlockArgument):
            return ("arg", func_args.index(value)) if value in func_args else ("arg", None)
        owner = value.owner
        oname = getattr(owner, "name", "")
        if value in result_owner:
            return ("op", result_owner[value][0])
        if oname in ("tensor.empty", "linalg.fill", "tensor.splat"):
            return ("init", oname.split(".")[-1])
        if oname == "arith.constant":
            return ("const", None)
        return ("other", oname)

    def _operand_rec(value):
        rec = {"source": _source(value), "shape": _shape(value.type), "dtype": _dtype(value.type)}
        if value in result_owner:
            # WHICH result of the producing op this edge reads. 0 for the single-result ops that are
            # the overwhelming majority; the disambiguator for a multi-result producer.
            rec["result_index"] = result_owner[value][1]
        cv = _scalar_const(value)
        if cv is not None:
            rec["const_value"] = cv
        return rec

    ops_out: list[dict[str, Any]] = []
    for i, op in enumerate(payload):
        ins, outs = _ins_outs(op)
        prov = _prov(op)
        rec: dict[str, Any] = {
            "id": i,
            "kind": op.name,
            "op": prov.get("op", op.name.split(".")[-1]),
            "family": prov.get("family", ""),
            "prov": prov,
            "ins": [_operand_rec(v) for v in ins],
            "outs": [_operand_rec(v) for v in outs],
            "results": [{"shape": _shape(r.type), "dtype": _dtype(r.type)} for r in op.results],
            "body_ops": _body_op_names(op),
            "reduction_dims": _reduction_dims(op),
        }
        ext = {}
        if op.name in _MATMUL_KINDS or prov.get("family") == "contraction":
            ext = _matmul_extents(ins, outs)
        if ext:
            rec["extents"] = ext
        ops_out.append(rec)

    return {
        "level": LEVEL,
        "entry": entry,
        "args": [{"index": i, "shape": _shape(a.type), "dtype": _dtype(a.type)} for i, a in enumerate(func_args)],
        "results": [{"shape": _shape(t), "dtype": _dtype(t)} for t in _fn_result_types(fn)],
        "ops": ops_out,
    }


def _fn_name(fn) -> str:
    from xdsl.dialects.builtin import StringAttr

    for table in (getattr(fn, "properties", {}) or {}, fn.attributes):
        v = table.get("sym_name")
        if isinstance(v, StringAttr):
            return v.data
    return "forward"


def _fn_result_types(fn) -> list:
    """The @forward result types, read from the function type."""
    ft = None
    for table in (getattr(fn, "properties", {}) or {}, fn.attributes):
        if "function_type" in table:
            ft = table["function_type"]
            break
    if ft is not None and hasattr(ft, "outputs"):
        return list(ft.outputs.data)
    # fallback: the terminator's operand types
    block = fn.body.blocks[0]
    term = block.ops.last
    return [o.type for o in term.operands] if term is not None else []


def matmul_records(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Convenience view: just the contraction (matmul-family) op records, with extents.

    A package that only lowers the matmul family (reusing the residency command path) filters here.
    """
    return [o for o in parsed.get("ops", []) if o["kind"] in _MATMUL_KINDS or o.get("family") == "contraction"]
