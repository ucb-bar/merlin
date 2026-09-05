"""Bind a capture's quantized-subclass inner tensors on the COMPILED path, not only in the
interpreter.

A torchao weight-only / W8A8 capture does not hand the int8 weight to `@forward` as an argument.
The subclass's inner tensors (`tensor_impl.int_data`, `tensor_impl.scale`) are extracted to
``extra.npz`` under ``qinner::<fqn>`` keys, and m2m leaves an **uninitialized** `tensor.empty` in
their place, tagging the CONSUMER with ``prov.quant_inner_<operand>`` (xDSL's printer drops
attributes on `tensor.empty`, so the key cannot live on the empty itself).

:func:`merlin.runtime.dispatch_runtime.execute` binds those empties at eval time from the npz. The
compiled path had no equivalent: `c_runtime.generate` builds its argument table from the `@forward`
signature, the tensors are not arguments, and the empties reach the binary as whatever the allocator
last left there. The result is the worst shape a defect can take -- the interpreter gates `cos 1.0`
on a bundle whose compiled classifier head reads garbage, and the two paths disagree with nothing
reporting it.

This module closes that gap in the compiler rather than in one bundle:

* :func:`lift` rewrites the module so each tagged empty becomes a trailing `@forward` argument
  (run from the shared preparation step, so every whole-model backend gets it);
* :func:`plan` derives the SAME ordered argument list from the same `model.mlir` without mutating
  anything, so :func:`merlin.llvmlower.c_runtime.generate` -- which re-parses the bundle, not the
  prepared IR -- appends matching rows and the real bytes. Both sides are one function of one input,
  which is what keeps the object and the ABI table from describing different things.

FAIL CLOSED. A tagged empty whose ``qinner::`` key is absent from ``extra.npz``, or whose stored
shape/dtype disagrees with the IR, is an error. Substituting zeros would reproduce exactly the
silent-garbage failure this exists to remove.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

#: prefix the capture uses for the extracted inner tensors inside ``extra.npz``
EXTRA_PREFIX = "qinner::"

#: ops that only re-view a tensor: a tagged empty reaches its consumer through them, so the tag
#: walk (and the uninitialized-read check) has to see through them rather than stop at them.
VIEW_OPS: frozenset[str] = frozenset({
    "tensor.cast", "tensor.collapse_shape", "tensor.expand_shape", "tensor.reshape",
    "tensor.extract_slice", "tensor.bitcast",
})


class QinnerError(RuntimeError):
    """A quant-inner tensor cannot be bound soundly. Never downgraded to a warning."""


@dataclass(frozen=True)
class QinnerArg:
    """One inner tensor lifted to a trailing `@forward` argument."""

    key: str                     #: the ``extra.npz`` key, without the ``qinner::`` prefix
    shape: tuple[int, ...]
    dtype: str                   #: MLIR element-type spelling (``i8``, ``f32``, ...)

    def as_json(self) -> dict[str, Any]:
        return {"key": self.key, "shape": list(self.shape), "dtype": self.dtype}

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "QinnerArg":
        return QinnerArg(str(obj["key"]), tuple(int(d) for d in obj["shape"]), str(obj["dtype"]))


def _elem_str(t) -> str:
    from xdsl.dialects.builtin import TensorType

    et = t.element_type if isinstance(t, TensorType) else t
    return str(et)


def _shape_of(t) -> tuple[int, ...]:
    return tuple(int(d) for d in t.get_shape())


def _op_name(op) -> str:
    """The op's dialect name, or the ``op_name`` provenance spelling when a generic carries one."""
    for attr in ("op_name__", "op_name"):
        if attr in getattr(op, "attributes", {}):
            spelling = getattr(op.attributes[attr], "data", None)
            if spelling:
                return str(spelling)
    return op.name


def _is_view(op) -> bool:
    return op.name in VIEW_OPS or _op_name(op) in VIEW_OPS


def tagged_empties(module) -> list[tuple[Any, str]]:
    """``[(tensor.empty op, qinner key)]`` in module order, deduplicated by op.

    Requires the tags to have been propagated onto the empties already
    (:func:`merlin.runtime.dispatch_runtime._propagate_quant_inner`); :func:`plan` and :func:`lift`
    both do that first, so callers do not have to remember the ordering.
    """
    found: list[tuple[Any, str]] = []
    seen: set[int] = set()
    for op in module.walk():
        if op.name != "tensor.empty":
            continue
        tag = op.attributes.get("prov.quant_inner")
        key = getattr(tag, "data", None)
        if key is None or id(op) in seen:
            continue
        seen.add(id(op))
        found.append((op, str(key)))
    return found


def _propagate(module) -> None:
    from ..runtime.dispatch_runtime import _propagate_quant_inner

    _propagate_quant_inner(module)


def plan(module) -> list[QinnerArg]:
    """The ordered argument list :func:`lift` would append, without mutating ``module``."""
    _propagate(module)
    return [QinnerArg(key, _shape_of(op.results[0].type), _elem_str(op.results[0].type))
            for op, key in tagged_empties(module)]


def plan_for_bundle(mlir_path: str | Path) -> list[QinnerArg]:
    """:func:`plan` for a bundle's ``model.mlir``, ``[]`` when it has no quantized subclass.

    Parsing a whole module costs seconds and almost no bundle carries a subclass, so the provenance
    key is looked for in the file's BYTES first and only a hit pays for a parse. That pre-check is a
    substring test on the key the capture writes -- not a pattern that could go stale against a
    differently spelled operand index -- and it can only ever skip a parse that would have found
    nothing."""
    from ..frontends.linalg_mlir import parse_mlir_file

    if b"prov.quant_inner" not in Path(mlir_path).read_bytes():
        return []
    return plan(parse_mlir_file(mlir_path))


def _entry_func(op):
    """The `func.func` that (transitively) contains ``op``."""
    cur = op
    while cur is not None:
        if getattr(cur, "name", None) == "func.func":
            return cur
        cur = getattr(cur, "parent_op", lambda: None)()
    raise QinnerError("quant-inner tensor.empty is not inside a func.func")


def lift(module) -> list[QinnerArg]:
    """Replace every quant-inner-tagged `tensor.empty` with a trailing `@forward` argument.

    Returns the appended arguments in argument order. A module with no tags is left untouched and
    returns ``[]`` -- so a bundle that never had a quantized subclass lowers byte-identically.
    """
    from xdsl.dialects.builtin import FunctionType

    _propagate(module)
    targets = tagged_empties(module)
    if not targets:
        return []

    funcs = {id(_entry_func(op)): _entry_func(op) for op, _ in targets}
    if len(funcs) != 1:
        raise QinnerError(
            f"quant-inner tensors span {len(funcs)} functions; expected one entry function")
    func = next(iter(funcs.values()))
    block = func.body.blocks[0]

    appended: list[QinnerArg] = []
    for op, key in targets:
        result = op.results[0]
        arg = block.insert_arg(result.type, len(block.args))
        result.replace_all_uses_with(arg)
        appended.append(QinnerArg(key, _shape_of(result.type), _elem_str(result.type)))
        op.detach()
        op.erase()
    func.properties["function_type"] = FunctionType.from_lists(
        [a.type for a in block.args], list(func.function_type.outputs.data))
    return appended


# --- the seam: writing / reading the plan beside the prepared IR ---------------------------------

#: file name the preparation step writes beside ``model.prepared.mlir``
PLAN_FILE = "qinner_args.json"


def write_plan(path: str | Path, args: Sequence[QinnerArg]) -> Path:
    path = Path(path)
    path.write_text(json.dumps([a.as_json() for a in args], indent=2) + "\n", encoding="utf-8")
    return path


def read_plan(path: str | Path) -> list[QinnerArg]:
    return [QinnerArg.from_json(o) for o in json.loads(Path(path).read_text(encoding="utf-8"))]


# --- binding the bytes --------------------------------------------------------------------------

#: MLIR element spelling -> the numpy dtype the npz must hold. An unknown spelling is an error, not
#: a reinterpretation of the stored bytes.
_NP_OF = {"i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64", "i1": "bool",
          "f16": "float16", "f32": "float32", "f64": "float64"}


def resolve(extra: Any, args: Iterable[QinnerArg]) -> list["Any"]:
    """The contiguous arrays for ``args``, read from an opened ``extra.npz``-like mapping.

    Raises :class:`QinnerError` when a key is missing or its stored shape/dtype disagrees with the
    IR -- never a zero substitute, which is the failure mode being removed.
    """
    import numpy as np

    files = set(getattr(extra, "files", None) or extra.keys())
    out = []
    for a in args:
        name = EXTRA_PREFIX + a.key
        if name not in files:
            raise QinnerError(
                f"quant-inner tensor {a.key!r} is tagged in the IR but absent from extra.npz "
                f"(looked for {name!r}); the compiled model would read uninitialized memory")
        arr = np.ascontiguousarray(extra[name])
        want_dt = _NP_OF.get(a.dtype)
        if want_dt is None:
            raise QinnerError(f"quant-inner tensor {a.key!r} has unsupported element type {a.dtype}")
        if tuple(int(d) for d in arr.shape) != a.shape or str(arr.dtype) != want_dt:
            raise QinnerError(
                f"quant-inner tensor {a.key!r} is {tuple(arr.shape)}x{arr.dtype} in extra.npz but "
                f"{a.shape}x{a.dtype} in the IR")
        out.append(arr)
    return out


# --- the gate: uninitialized data must not reach computation ------------------------------------

def uninitialized_reads(module) -> list[str]:
    """Every `tensor.empty` value that is READ by a compute op, as human-readable findings.

    A `tensor.empty` is a destination-passing-style *destination*: legitimate uses are an `outs`/init
    operand, which the consumer overwrites. A use as an `ins` operand -- or as a returned value --
    means the model computes on memory nobody wrote, which is exactly what an unbound quant-inner
    tensor looks like after codegen. Derived structurally from the IR (linalg's own ins/outs operand
    split), so it is not specific to any model, quantization scheme or target.
    """
    # Propagate the capture's consumer-side tags onto the empties first, so a finding can NAME the
    # tensor that is unbound instead of only its type. Idempotent, and additive: it only writes
    # `prov.quant_inner` onto ops that a consumer already pointed at.
    _propagate(module)
    # id(SSAValue) -> (type spelling, the quant-inner key when the capture named one)
    tainted: dict[int, tuple[str, str | None]] = {}
    findings: list[str] = []

    def _reads(op) -> list[Any]:
        """Operands whose ELEMENTS the op reads (not destinations, not shape-only operands)."""
        inputs = getattr(op, "inputs", None)
        if inputs is not None and getattr(op, "outputs", None) is not None:
            inputs = list(inputs)
            # A structured op's `ins` operands are its body's leading block arguments. An operand
            # whose block argument is UNUSED contributes only its shape -- the window extent of a
            # pooling op is carried exactly that way (`aten.max_pool2d` lowers to a generic with an
            # unread KxK operand, as upstream `linalg.pooling_*` does), and flagging it would refuse
            # every pooling model. Read the body, do not special-case the op name.
            body = op.regions[0].blocks[0] if op.regions and op.regions[0].blocks else None
            if body is not None and len(body.args) >= len(inputs):
                return [v for j, v in enumerate(inputs) if any(True for _ in body.args[j].uses)]
            return inputs
        if op.name in ("func.return", "scf.yield"):
            return list(op.operands)
        if _is_view(op):
            return list(op.operands[:1])
        return []

    for op in module.walk():
        if op.name == "tensor.empty":
            tag = op.attributes.get("prov.quant_inner")
            tainted[id(op.results[0])] = (str(op.results[0].type), getattr(tag, "data", None))
            continue
        reads = _reads(op)
        for operand in reads:
            origin = tainted.get(id(operand))
            if origin is None:
                continue
            if _is_view(op):
                # a view of uninitialized data is still uninitialized: carry the taint forward
                for result in op.results:
                    tainted[id(result)] = origin
                continue
            spelling, key = origin
            findings.append(
                f"{op.name} reads an uninitialized tensor.empty of type {spelling}"
                + (f" (quant-inner key {key!r}, unbound)" if key else ""))
    return findings


def require_initialized(module, *, where: str = "") -> None:
    """Raise unless no uninitialized `tensor.empty` is read. Called from the shared preparation
    step, so a bundle whose compiled binary would compute on unwritten memory fails the BUILD
    instead of producing a plausible number."""
    findings = uninitialized_reads(module)
    if not findings:
        return
    head = f"{where}: " if where else ""
    detail = "\n  ".join(sorted(set(findings))[:20])
    raise QinnerError(
        f"{head}{len(findings)} uninitialized tensor(s) reach computation; the compiled binary "
        f"would read unwritten memory while the numpy interpreter binds them:\n  {detail}\n"
        "If these are quantized-subclass inner tensors, the capture must tag them "
        "(prov.quant_inner_<operand>) and store them in extra.npz under a qinner:: key.")
