"""Structured reads over model2MLIR (linalg-on-tensors) modules via xDSL — no regex on IR text.

Several subsystems used to scrape facts out of MLIR *text* — ``func.func @forward(...)`` signatures,
``tensor<...>`` shapes, op names, ``prov.*`` attributes — with regex. That is brittle and drifts from
the IR's real structure. This module is the one grounded query surface: it parses through
:func:`merlin.frontends.linalg_mlir.parse_mlir_text` (the same xDSL path proven on real smolvla/pi05
captures — Builtin/Func/Arith/Linalg/Tensor/Scf/Math/Cf, ``allow_unregistered``, plus the
``} -> (T1,T2)`` normalizer) and exposes small typed accessors over the parsed module.

Custom-dialect ops (``quant_ext.*``, a target's own dialect, anything model2MLIR emits) round-trip as xDSL
``UnregisteredOp`` — :func:`op_name` recovers their real name, and attribute reads work unchanged.

xDSL-gated: requires the ``xdsl`` install (present in the default .venv). No ``re`` here by design.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import RLock
from typing import Any

from merlin.frontends.linalg_mlir import _dtype as _type_dtype
from merlin.frontends.linalg_mlir import _shape as _type_shape
from merlin.frontends.linalg_mlir import parse_mlir_file, parse_mlir_text

# Parsing a big capture (tens of thousands of ops) costs seconds; the DSE analysis path reads the
# same model.mlir through several functions (numerical_contract, attribution, models, quant_rows,
# capture_erasure), so memoize by resolved path and actual source bytes. Modules are read-only for every
# mlir_query consumer, so sharing one parsed instance is safe. Bounded LRU to cap memory.
_PARSE_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_PARSE_CACHE_MAX = 16
_PARSE_CACHE_LOCK = RLock()
_PARSE_SCOPE: ContextVar[tuple | None] = ContextVar("mlir_query_parse_scope", default=None)
_SIGNATURE_CACHE: OrderedDict = OrderedDict()
_SIGNATURE_CACHE_MAX = 32
_SIGNATURE_CACHE_LOCK = RLock()


@contextmanager
def parse_scope(*, max_entries=4):
    """Bound file-parse reuse to one ordinary invocation; nested callers share it.

    Every lookup still reads exact current bytes and checks them after parsing
    or a hit. Returned modules follow parse's existing read-only convention;
    mutation consumers must clone. No global cache entry is borrowed, and all
    local graph references are released at the outer scope's exit.
    """
    if type(max_entries) is not int or not 1 <= max_entries <= _PARSE_CACHE_MAX:
        raise ValueError("parse scope requires an explicit bounded file-graph budget")
    if _PARSE_SCOPE.get() is not None:
        yield
        return
    local = OrderedDict()
    token = _PARSE_SCOPE.set((local, max_entries))
    try:
        yield
    finally:
        local.clear()
        _PARSE_SCOPE.reset(token)


def parse(src: Any, *, source_text: str | None = None):
    """Parse ``src`` into an xDSL module. ``src`` may be an already-parsed module, MLIR text, or a
    path to an ``.mlir`` file (a short, newline-free string that names an existing file). File
    parses are memoized by content (timestamps alone never establish correspondence)."""
    # An already-parsed xDSL module (not a str/Path — note pathlib.Path grew a .walk() in 3.13, so a
    # bare hasattr(src, "walk") would misclassify a path as a module).
    if not isinstance(src, (str, Path)) and hasattr(src, "walk"):
        return src
    s = str(src)
    if "\n" not in s and len(s) < 4096:
        p = Path(s)
        if p.is_file() or source_text is not None:
            payload = p.read_bytes() if source_text is None else source_text.encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            key = (str(p.resolve()), digest)
            scope = _PARSE_SCOPE.get()
            cache, maximum = scope if scope is not None else (_PARSE_CACHE, _PARSE_CACHE_MAX)
            with _PARSE_CACHE_LOCK:
                hit = cache.get(key)
                if hit is not None:
                    cache.move_to_end(key)
            if hit is not None:
                if source_text is None and hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"MLIR source changed during cached parse read: {p}")
                return hit
            # Through the FILE door, which falls back to a generic re-print when MLIR printed the
            # module in custom form. Reading the bytes here and handing them to the TEXT parser
            # skipped that fallback, and this is the consumer where skipping it is silent: the
            # caller (`perop_blocks._observe_conv_contractions`) turns any parse failure into "no
            # contractions observed" and drops the register block, so `perop_register_block` was
            # reported as applied and did nothing on every model whose prepared module had been
            # round-tripped by `prov_cse` / `perop_blocks` -- measured: all three of lstmnetvit,
            # resnet50_v1_5 and tiny_llama.
            module = parse_mlir_file(p, source_text=payload.decode("utf-8"))
            if source_text is None and hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                raise ValueError(f"MLIR source changed during parsing: {p}")
            with _PARSE_CACHE_LOCK:
                cache[key] = module
                cache.move_to_end(key)
                while len(cache) > maximum:
                    cache.popitem(last=False)
            return module
    return parse_mlir_text(s)


def op_name(op) -> str:
    """The op's real name, recovering the original for xDSL ``UnregisteredOp`` (so a generic-form
    ``"quant_ext.dequantize_per_channel"(...)`` reports that, not ``builtin.unregistered``)."""
    if op.name == "builtin.unregistered":
        real = getattr(op, "op_name", None)
        if real is not None:
            return getattr(real, "data", str(real))
    return op.name


def walk(module, *names: str) -> Iterator:
    """Iterate ops in ``module``; if ``names`` are given, only ops whose real :func:`op_name` matches
    (registered or unregistered)."""
    wanted = set(names)
    for op in module.walk():
        if not wanted or op_name(op) in wanted:
            yield op


def op_count(module, *names: str) -> int:
    """Count ops whose real name is in ``names`` (or all ops if no names given)."""
    return sum(1 for _ in walk(module, *names))


def type_shape_dtype(t) -> tuple[list[int], str]:
    """``(shape, dtype)`` for a tensor type — ``([], str(t))`` for a non-tensor/scalar type."""
    return list(_type_shape(t)), _type_dtype(t)


def _sym_name(fn) -> str | None:
    sym = getattr(fn, "sym_name", None)
    return getattr(sym, "data", None) if sym is not None else None


def _closing(text: str, start: int, opener: str, closer: str) -> int:
    """Index just past the bracket that closes ``text[start]`` (an ``opener``), skipping string literals;
    ``-1`` when it never closes."""
    depth, i, n = 0, start, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _signature_skeleton(text: str, func_name: str) -> str | None:
    """A module holding only ``@func_name``'s printed header over an empty body, or ``None`` when the
    header is not in the custom ``func.func @name(...)`` form this reader handles.

    The function type is fully stated by the header (its arguments and its ``->`` results); a
    transformer's body is ~100k lines a full parse spends a minute on to read a few hundred types.
    """
    marker = f"@{func_name}("
    at = 0
    while True:
        at = text.find(marker, at)
        if at < 0:
            return None
        line_start = text.rfind("\n", 0, at) + 1
        words = text[line_start:at].split()
        if words and words[0] == "func.func":
            break
        at += 1
    end = _closing(text, at + len(marker) - 1, "(", ")")
    if end < 0:
        return None
    # After the arguments: optional results, optional `attributes {...}`, then the body's `{`.
    i = end
    while i < len(text):
        if text.startswith("attributes", i) and text[i + len("attributes") :].lstrip().startswith("{"):
            brace = text.index("{", i)
            i = _closing(text, brace, "{", "}")
            if i < 0:
                return None
            continue
        if text[i] in "(<[":
            closer = {"(": ")", "<": ">", "[": "]"}[text[i]]
            i = _closing(text, i, text[i], closer)
            if i < 0:
                return None
            continue
        if text[i] == "{":
            return "builtin.module {\n" + text[line_start:i] + "{\n    func.return\n  }\n}\n"
        i += 1
    return None


def _copy_signature(value):
    return tuple([(list(shape), dtype) for shape, dtype in fields] for fields in value)


def forward_signature(
    src: Any, func_name: str = "forward"
) -> tuple[list[tuple[list[int], str]], list[tuple[list[int], str]]]:
    """Read a typed function signature, caching immutable file-byte snapshots.

    Header-only reads and full-module fallback consume the SAME source snapshot.
    File cache keys include actual content, not timestamps; callers receive fresh
    shape lists. Concurrent source mutation fails closed before publication.
    """
    key, path, text = None, None, None
    if isinstance(src, (str, Path)):
        s = str(src)
        is_path = "\n" not in s and len(s) < 4096 and Path(s).is_file()
        if is_path:
            path = Path(s)
            payload = path.read_bytes()
            text = payload.decode("utf-8")
            key = (str(path.resolve()), hashlib.sha256(payload).hexdigest(), func_name)
            with _SIGNATURE_CACHE_LOCK:
                cached = _SIGNATURE_CACHE.get(key)
                if cached is not None:
                    _SIGNATURE_CACHE.move_to_end(key)
            if cached is not None:
                if hashlib.sha256(path.read_bytes()).hexdigest() != key[1]:
                    raise ValueError(f"MLIR source changed during signature read: {path}")
                return _copy_signature(cached)
        else:
            text = s

    def finish(result):
        if key is None:
            return result
        if hashlib.sha256(path.read_bytes()).hexdigest() != key[1]:
            raise ValueError(f"MLIR source changed during signature parsing: {path}")
        immutable = tuple(tuple((tuple(shape), dtype) for shape, dtype in fields) for fields in result)
        with _SIGNATURE_CACHE_LOCK:
            _SIGNATURE_CACHE[key] = immutable
            _SIGNATURE_CACHE.move_to_end(key)
            while len(_SIGNATURE_CACHE) > _SIGNATURE_CACHE_MAX:
                _SIGNATURE_CACHE.popitem(last=False)
        return _copy_signature(immutable)

    if text is not None:
        skeleton = _signature_skeleton(text, func_name)
        if skeleton is not None:
            try:
                fn = next(op for op in parse_mlir_text(skeleton).walk() if op.name == "func.func")
                ftype = fn.function_type
            except Exception:  # noqa: BLE001 -- full typed parse remains the fallback definition
                pass
            else:
                return finish(
                    (
                        [type_shape_dtype(t) for t in ftype.inputs.data],
                        [type_shape_dtype(t) for t in ftype.outputs.data],
                    )
                )
    module = parse(src, source_text=text) if path is not None else parse(src)
    fn = next((op for op in module.walk() if op.name == "func.func" and _sym_name(op) == func_name), None)
    if fn is None:
        raise ValueError(f"no func.func @{func_name} in module")
    ftype = fn.function_type
    return finish(([type_shape_dtype(t) for t in ftype.inputs.data], [type_shape_dtype(t) for t in ftype.outputs.data]))


#: dtype string -> bytes per element, for the dtypes model2MLIR captures emit.
_DTYPE_BYTES = {"f64": 8, "f32": 4, "f16": 2, "bf16": 2, "i64": 8, "i32": 4, "i16": 2, "i8": 1, "i1": 1}


def value_bytes(t) -> int:
    """Footprint of one SSA value of tensor type ``t``; 0 for scalars and dynamic shapes."""
    shape, dtype = type_shape_dtype(t)
    width = _DTYPE_BYTES.get(dtype)
    if not shape or width is None or any(d < 0 for d in shape):
        return 0
    n = 1
    for d in shape:
        n *= d
    return n * width


def activation_peak_bytes(src: Any, func_name: str = "forward") -> int | None:
    """Peak SIMULTANEOUSLY-LIVE intermediate bytes of ``@func_name`` — the model's working set.

    Every captured tensor is statically shaped, so a value's footprint is exact and its live range is
    [defining op, last use]; the peak over program points is the activation memory the run needs.
    Function arguments are excluded: weights and inputs are bound from the weights blob / arg table,
    not allocated. Nested-region ops (an ``scf.for`` body) count as their results only, which is what
    the caller wants — a loop's internal temporaries are freed within it.

    This is a LOWER BOUND on the runtime arena: bufferization introduces copies the tensor form does
    not show, and the allocator fragments. Callers must add slack, not treat it as the requirement.

    Returns ``None`` when it cannot be measured (unparseable module, dynamic shapes, no such
    function) so a caller can fall back rather than provision from a wrong number.
    """
    try:
        module = parse(src)
        fn = next((op for op in module.walk() if op.name == "func.func" and _sym_name(op) == func_name), None)
        if fn is None or not fn.body.blocks:
            return None
        ops = list(fn.body.blocks[0].ops)
        last_use: dict[Any, int] = {}
        for i, op in enumerate(ops):
            for operand in op.operands:
                last_use[operand] = i
        live: dict[Any, int] = {}
        peak = 0
        for i, op in enumerate(ops):
            for res in op.results:
                nbytes = value_bytes(res.type)
                if nbytes:
                    live[res] = nbytes
            peak = max(peak, sum(live.values()))
            for val in [v for v in live if last_use.get(v, -1) <= i]:
                del live[val]
        return peak or None
    except Exception:  # noqa: BLE001
        return None


def _attr_tables(op):
    """Both attribute stores an xDSL op may use (dialect ops put custom attrs in ``properties``;
    unregistered ops keep them in ``attributes``)."""
    yield op.attributes
    props = getattr(op, "properties", None)
    if props:
        yield props


def attr_str(op, key: str) -> str | None:
    """String value of attribute ``key`` on ``op`` (StringAttr ``.data``, else stringified without
    surrounding quotes), searching both the attribute and property stores; None if absent."""
    for table in _attr_tables(op):
        v = table.get(key)
        if v is None:
            continue
        data = getattr(v, "data", None)
        return data if isinstance(data, str) else str(v).strip().strip('"')
    return None


def provenance(op) -> dict[str, str]:
    """All ``prov.*`` string attributes on ``op`` (from either attribute store)."""
    out: dict[str, str] = {}
    for table in _attr_tables(op):
        for key in table:
            if key.startswith("prov.") and key not in out:
                val = attr_str(op, key)
                if val is not None:
                    out[key] = val
    return out
