"""Whole-model compilability: what does the compiler make of a REAL model2MLIR linalg module?

The staged pipeline (``lower_module``) and the Python engine model a specific vocabulary — matmul,
elementwise add/mul, relu, sum-reduction, per-channel weight dequant, pure layout reshapes. A real
transformer (tiny_llama, smolVLA, …) also carries rmsnorm, softmax, and rope, whose bodies need
``rsqrt``/``exp``/``div``/``sin``/``cos`` — math the engine does not implement. Rather than lower such
a model and crash on the first unmodeled op, this pass walks the parsed module and reports, op by op,
what is engine-expressible today, what is merely plumbing-not-yet-written, and what is a fundamental
gap — plus the matmul backbone inventory with the real shapes and the int8 weight-only dequant idiom
resolved (weight traced through ``dequantize -> transpose`` back to its function argument).

Classification is structural (op names, ``prov.family``, and a scan of each region body) — no regex,
no target names. It is derived from the module, never assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._common import HAS_XDSL

# Op categories. MODELED_ENGINE = the math is in the engine's vocabulary; PIPELINE_TODAY = a subset
# `lower_module` already lowers end-to-end (the rest is plumbing for a later increment).
MODELED_ENGINE = {"matmul", "elementwise", "activation", "reduction", "view", "quant", "scaffold"}
PIPELINE_TODAY = {"matmul", "elementwise", "activation", "scaffold"}

# Body math that the engine has no operator for — its presence makes an op a FUNDAMENTAL gap
# (needs a new engine primitive), not merely un-plumbed lowering.
_UNMODELED_MATH = {"math.rsqrt", "math.exp", "math.powf", "math.sin", "math.cos",
                   "math.log", "math.sqrt", "math.tanh", "math.erf", "arith.divf"}
# Body math the engine DOES model elementwise.
_MODELED_MATH = {"arith.mulf", "arith.addf", "arith.subf", "arith.maximumf", "linalg.yield",
                 "arith.muli", "arith.addi"}

_VIEW_OPS = {"tensor.expand_shape", "tensor.collapse_shape", "tensor.cast", "tensor.extract_slice",
             "tensor.concat", "linalg.transpose", "linalg.copy", "tensor.extract", "tensor.insert"}
_SCAFFOLD_OPS = {"tensor.empty", "arith.constant", "tensor.splat", "linalg.yield", "linalg.index",
                 "arith.index_cast", "func.return", "linalg.fill"}
_MATMUL_OPS = {"linalg.matmul", "linalg.batch_matmul", "linalg.quantized_matmul"}
# Named linalg elementwise / activation ops (as opposed to the linalg.generic bodies the real
# model emits) — the engine's vector path models these directly.
_LINALG_ELEMENTWISE = {"linalg.add", "linalg.mul", "linalg.sub"}
_LINALG_ACTIVATION = {"linalg.max"}


def _op_name(op) -> str:
    """Resolved op name — model2MLIR's ``quant_ext`` ops parse as ``builtin.unregistered`` with the
    true name on the ``op_name`` field; everything else uses ``op.name``."""
    if op.name == "builtin.unregistered":
        on = getattr(op, "op_name", None)
        return on.data if on is not None else op.name
    return op.name


def _prov(op) -> dict[str, str]:
    from xdsl.dialects.builtin import StringAttr

    out: dict[str, str] = {}
    for key, val in op.attributes.items():
        if key.startswith("prov.") and isinstance(val, StringAttr):
            out[key] = val.data
    return out


def _body_op_names(op) -> set[str]:
    """Resolved op names inside an op's first region body (empty for ops with no region)."""
    names: set[str] = set()
    for region in op.regions:
        for block in region.blocks:
            for inner in block.ops:
                names.add(_op_name(inner))
    return names


def _has_reduction_iterator(op) -> bool:
    """True when a linalg op declares a reduction iterator (structurally, from its attribute)."""
    it = op.attributes.get("iterator_types")
    if it is None:
        return False
    for entry in it:
        # xDSL linalg iterator entries stringify to include "reduction" for a reduced axis.
        if "reduction" in str(entry):
            return True
    return False


def classify_op(op) -> str:
    """One of MODELED_ENGINE ∪ {"unmodeled"}. Structural: name + prov.family + body scan."""
    name = _op_name(op)
    if name in _SCAFFOLD_OPS:
        return "scaffold"
    if name in _VIEW_OPS:
        return "view"
    if name.startswith("quant_ext.dequantize") or name.startswith("quant_ext.quantize"):
        return "quant"
    if name in _MATMUL_OPS:
        return "matmul"
    if name in _LINALG_ELEMENTWISE:
        return "elementwise"
    if name in _LINALG_ACTIVATION:
        return "activation"
    # arith/math scalars appearing at the top level (rare) classify by name.
    if name in _UNMODELED_MATH:
        return "unmodeled"
    if name in _MODELED_MATH:
        return "elementwise"
    if name in ("linalg.generic", "linalg.reduce"):
        body = _body_op_names(op)
        if body & _UNMODELED_MATH:
            return "unmodeled"
        fam = _prov(op).get("prov.family", "")
        if name == "linalg.reduce" or _has_reduction_iterator(op):
            # The engine reduces by sum only; a body that is pure addf is a sum-reduction.
            return "reduction" if body <= _MODELED_MATH | {"arith.maximumf"} else "unmodeled"
        if fam in ("elementwise", "fill"):
            return "elementwise"
        if body and body <= _MODELED_MATH:
            return "elementwise"
        return "unmodeled"
    return "unmodeled"


def _trace_weight(value, func_args):
    """Trace a matmul RHS back through the int8 weight-only idiom (dequantize -> transpose ->
    reshape) to its function argument. Returns (weight_arg_index, scale_arg_index, quant_op_name,
    weight_dtype) — any component is None when it cannot be resolved (fail-open on the trace, the
    caller records what it got)."""
    from xdsl.ir import BlockArgument

    scale_idx = None
    quant = None
    dtype = None
    seen = 0
    while seen < 32:
        seen += 1
        if isinstance(value, BlockArgument):
            idx = func_args.index(value) if value in func_args else None
            return idx, scale_idx, quant, dtype
        owner = getattr(value, "owner", None)
        nm = _op_name(owner) if owner is not None else ""
        if nm in ("tensor.expand_shape", "tensor.collapse_shape", "tensor.cast", "linalg.copy"):
            value = owner.operands[0]
        elif nm == "linalg.transpose":
            value = owner.inputs[0]
        elif nm.startswith("quant_ext.dequantize"):
            quant = nm
            dtype = _elem_dtype(owner.operands[0].type)
            if len(owner.operands) > 1:                     # per-channel scale operand
                s = owner.operands[1]
                if isinstance(s, BlockArgument) and s in func_args:
                    scale_idx = func_args.index(s)
            value = owner.operands[0]
        else:
            return None, scale_idx, quant, dtype
    return None, scale_idx, quant, dtype


def _shape(t) -> tuple[int, ...]:
    try:
        return tuple(t.get_shape())
    except Exception:  # pragma: no cover - non-tensor operand
        return ()


def _elem_dtype(t) -> str:
    from xdsl.dialects.builtin import IntegerType

    elem = getattr(t, "element_type", None)
    if isinstance(elem, IntegerType):
        return "i%d" % elem.width.data
    return str(elem) if elem is not None else "?"


@dataclass
class MatmulSite:
    region_id: str | None
    m: int | None
    k: int | None
    n: int | None
    lhs_shape: tuple[int, ...]
    weight_arg: int | None
    scale_arg: int | None
    quant: str | None
    weight_dtype: str | None


@dataclass
class CompilabilityReport:
    matmuls: list[MatmulSite]
    op_classes: dict[str, int]
    blockers: list[dict[str, Any]]
    modeled: bool           # every op is engine-expressible (no fundamental gap)
    pipeline_ready: bool    # every op is one lower_module lowers end-to-end today
    unmodeled_families: dict[str, int] = field(default_factory=dict)


def compilability_report(module) -> CompilabilityReport:
    """Walk a parsed linalg module's function body and report what the compiler makes of it."""
    if not HAS_XDSL:
        raise RuntimeError("xDSL is required for the compilability report")
    fns = [op for op in module.walk() if op.name == "func.func"]
    if not fns:
        raise ValueError("no func.func in module")
    fn = fns[0]
    func_args = list(fn.body.blocks[0].args)

    op_classes: dict[str, int] = {}
    blockers: list[dict[str, Any]] = []
    unmodeled_families: dict[str, int] = {}
    matmuls: list[MatmulSite] = []

    for op in fn.body.blocks[0].ops:
        cat = classify_op(op)
        op_classes[cat] = op_classes.get(cat, 0) + 1
        name = _op_name(op)
        prov = _prov(op)
        if cat == "matmul":
            lhs, rhs = op.inputs[0], op.inputs[1]
            ls, rs = _shape(lhs.type), _shape(rhs.type)
            m = k = n = None
            if len(ls) == 2 and len(rs) == 2:
                m, k = ls
                _, n = rs
            w_idx, s_idx, quant, dtype = _trace_weight(rhs, func_args)
            matmuls.append(MatmulSite(
                region_id=prov.get("prov.region_id"), m=m, k=k, n=n, lhs_shape=ls,
                weight_arg=w_idx, scale_arg=s_idx, quant=quant, weight_dtype=dtype))
        if cat == "unmodeled":
            fam = prov.get("prov.family", "?")
            unmodeled_families[fam] = unmodeled_families.get(fam, 0) + 1
            blockers.append({"op": name, "region_id": prov.get("prov.region_id"),
                             "family": fam, "body_math": sorted(_body_op_names(op) & _UNMODELED_MATH)})

    modeled = op_classes.get("unmodeled", 0) == 0
    pipeline_ready = all(c in PIPELINE_TODAY for c in op_classes)
    return CompilabilityReport(matmuls=matmuls, op_classes=op_classes, blockers=blockers,
                               modeled=modeled, pipeline_ready=pipeline_ready,
                               unmodeled_families=unmodeled_families)


def report_from_file(path) -> CompilabilityReport:
    """Parse a model2MLIR linalg (+quant_ext) file and report its compilability."""
    from merlin.frontends import linalg_mlir as fl
    try:
        from merlin.frontends import quant_ext
        if quant_ext.available():
            return compilability_report(quant_ext.parse_quant_mlir(path))
    except Exception:
        pass
    return compilability_report(fl.parse_mlir_file(path))
