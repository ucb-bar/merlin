"""Serialize a composed CCA (the ``kernels.cca`` dataclass) to/from the ``cca`` MLIR dialect.

This is the bridge that makes "the CCA is expressed in MLIR" real: the deterministic analyzers produce
a ``CCA`` dataclass (from asm decode / flat-graph inspection), and ``to_mlir`` emits it as a ``cca.kernel``
MLIR artifact; ``from_mlir`` parses it back. Round-trippable for every scalar facet the compare/bijection
consume. Lives on the analysis side (``kernels``) so the dialect (``xdsl_dialects.cca``) never depends on
``kernels``. No LLM in this path — pure, deterministic (de)serialization.
"""
from __future__ import annotations

from .cca import CCA, ComputeFacet, VectorFacet

# Per-field coercion FROM the uniform StringAttr text back to the dataclass type. Strings pass through.
_BOOL = {"true": True, "false": False}


def _s(v) -> str | None:
    """Dataclass value -> attribute text (None -> omitted)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _compute_props(c: ComputeFacet) -> dict[str, str]:
    mr = c.register_block[0] if isinstance(c.register_block, (tuple, list)) and c.register_block else None
    fields = {
        "contraction_form": c.contraction_form, "accumulator_dtype": c.accumulator_dtype,
        "widening": c.widening, "reduction_form": c.reduction_form,
        "register_block_mr": mr, "epilogue": c.epilogue,
        "accumulator_resident": c.accumulator_resident, "nr_is_vsetvlmax": c.nr_is_vsetvlmax,
        "activation_vectorization": c.activation_vectorization,
    }
    return {k: _s(v) for k, v in fields.items() if _s(v) is not None}


def _vector_props(v: VectorFacet) -> dict[str, str]:
    fields = {"sew": v.sew, "lmul": v.lmul, "vl_strategy": v.vl_strategy, "tail": v.tail}
    return {k: _s(val) for k, val in fields.items() if _s(val) is not None}


def to_mlir(cca: CCA) -> str:
    """Emit a composed CCA as ``cca``-dialect MLIR text (a ``cca.kernel`` wrapping compute/vector ops)."""
    from xdsl.dialects.builtin import ModuleOp, StringAttr
    from xdsl.ir import Block, Region

    from ..xdsl_dialects import cca as D
    from ..xdsl_dialects._common import text

    def _props(d: dict[str, str]) -> dict:
        return {k: StringAttr(v) for k, v in d.items()}

    inner = [D.ComputeOp(properties=_props(_compute_props(cca.compute)))]
    if cca.vector is not None:
        inner.append(D.VectorOp(properties=_props(_vector_props(cca.vector))))
    kprops = {"op": cca.op, "backend": ",".join(cca.backend), "source": cca.provenance.get("source"),
              "level": cca.provenance.get("level")}
    kernel = D.KernelOp(properties=_props({k: v for k, v in kprops.items() if v}),
                        regions=[Region([Block(inner)])])
    return text(ModuleOp([kernel]))


def _get(op, field: str) -> str | None:
    a = getattr(op, field, None)
    return a.data if a is not None else None


def from_mlir(mlir_text: str) -> CCA:
    """Parse ``cca``-dialect MLIR text back into a CCA dataclass (scalar facets restored faithfully;
    register_block restored as ``(mr, None)`` — the symbolic VL-relative NR is not a scalar the compare
    uses, so it is intentionally not round-tripped)."""
    from xdsl.parser import Parser

    from ..xdsl_dialects import cca as D
    from ..xdsl_dialects._common import make_context

    module = Parser(make_context(D.get_dialect()), mlir_text).parse_module()
    kernel = next(o for o in module.body.block.ops if isinstance(o, D.KernelOp))
    compute_op = next((o for o in kernel.body.block.ops if isinstance(o, D.ComputeOp)), None)
    vector_op = next((o for o in kernel.body.block.ops if isinstance(o, D.VectorOp)), None)

    def _b(v):
        return _BOOL.get(v) if v is not None else None

    compute = ComputeFacet()
    if compute_op is not None:
        mr = _get(compute_op, "register_block_mr")
        compute = ComputeFacet(
            op=_get(kernel, "op"),
            contraction_form=_get(compute_op, "contraction_form"),
            accumulator_dtype=_get(compute_op, "accumulator_dtype"),
            widening=_b(_get(compute_op, "widening")),
            reduction_form=_get(compute_op, "reduction_form"),
            register_block=(int(mr), None) if mr is not None else None,
            epilogue=_get(compute_op, "epilogue"),
            accumulator_resident=_b(_get(compute_op, "accumulator_resident")),
            nr_is_vsetvlmax=_b(_get(compute_op, "nr_is_vsetvlmax")),
            activation_vectorization=_get(compute_op, "activation_vectorization"))

    vector = None
    if vector_op is not None:
        sew = _get(vector_op, "sew")
        lmul = _get(vector_op, "lmul")
        vector = VectorFacet(sew=int(sew) if sew is not None else None,
                             lmul=float(lmul) if lmul is not None else None,
                             vl_strategy=_get(vector_op, "vl_strategy"),
                             tail=_get(vector_op, "tail"))

    backend = _get(kernel, "backend")
    return CCA(op=_get(kernel, "op") or (compute.op or "unknown"),
               backend=backend.split(",") if backend else [],
               compute=compute, vector=vector,
               provenance={k: v for k, v in
                           (("source", _get(kernel, "source")), ("level", _get(kernel, "level")))
                           if v is not None})
