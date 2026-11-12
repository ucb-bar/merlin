"""Serialize a composed CCA (the ``kernels.cca`` dataclass) to/from the ``cca`` MLIR dialect.

This is the bridge that makes "the CCA is expressed in MLIR" real: the deterministic analyzers produce
a ``CCA`` dataclass (from asm decode / flat-graph inspection), and ``to_mlir`` emits it as a ``cca.kernel``
MLIR artifact; ``from_mlir`` parses it back. All analyzer facets, scope, tuple fields and provenance
round-trip without losing unknown values. Lives on the analysis side (``kernels``) so the dialect
(``xdsl_dialects.cca``) never depends on ``kernels``. No LLM in this path — pure, deterministic (de)serialization.
"""

from __future__ import annotations

import dataclasses
import json
import math
import types
import typing

from merlin.analysis.kernels.cca import CCA, ComputeFacet, MemoryFacet, VectorFacet

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
        "contraction_form": c.contraction_form,
        "accumulator_dtype": c.accumulator_dtype,
        "widening": c.widening,
        "reduction_form": c.reduction_form,
        "register_block_mr": mr,
        "epilogue": c.epilogue,
        "accumulator_resident": c.accumulator_resident,
        "nr_is_vsetvlmax": c.nr_is_vsetvlmax,
        "activation_vectorization": c.activation_vectorization,
    }
    return {k: _s(v) for k, v in fields.items() if _s(v) is not None}


def _vector_props(v: VectorFacet) -> dict[str, str]:
    fields = {"sew": v.sew, "lmul": v.lmul, "vl_strategy": v.vl_strategy, "tail": v.tail}
    return {k: _s(val) for k, val in fields.items() if _s(val) is not None}


def _memory_props(m) -> dict[str, str]:
    fields = {"access_pattern": m.access_pattern, "panel_reuse": m.panel_reuse, "a_broadcast_vf": m.a_broadcast_vf}
    return {k: _s(val) for k, val in fields.items() if _s(val) is not None}


def _facet_classes() -> dict:
    """Derive the serialization universe from the actual CCA field types."""
    result = {}
    for name, hint in typing.get_type_hints(CCA).items():
        choices = typing.get_args(hint) if typing.get_origin(hint) in (typing.Union, types.UnionType) else (hint,)
        for choice in choices:
            if isinstance(choice, type) and dataclasses.is_dataclass(choice):
                result[name] = choice
                break
    return result


def _json(value) -> str:
    def nested(item):
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            return dataclasses.asdict(item)
        raise TypeError("CCA data requires declared dataclass or JSON values")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=nested)


def _typed_value(value, hint):
    """Restore tuple identity and refuse malformed analyzer field values."""
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        for alternative in typing.get_args(hint):
            try:
                return _typed_value(value, alternative)
            except ValueError:
                pass
        raise ValueError("CCA facet value does not match its declared type")
    if hint is type(None):
        if value is None:
            return None
    elif isinstance(hint, type) and dataclasses.is_dataclass(hint):
        if isinstance(value, hint):
            value = dataclasses.asdict(value)
        hints = typing.get_type_hints(hint)
        if isinstance(value, dict) and set(value) == {f.name for f in dataclasses.fields(hint)}:
            return hint(**{key: _typed_value(item, hints[key]) for key, item in value.items()})
    elif origin is tuple or hint is tuple:
        if isinstance(value, (list, tuple)):
            args = typing.get_args(hint)
            if args and args[-1] is Ellipsis:
                return tuple(_typed_value(v, args[0]) for v in value)
            if args:
                if len(args) != len(value):
                    raise ValueError("CCA facet tuple length mismatch")
                return tuple(_typed_value(v, h) for v, h in zip(value, args, strict=True))
            return tuple(value)
    elif hint is float:
        if type(value) in (int, float) and math.isfinite(value):
            return float(value)
    elif hint in (str, int, bool) and type(value) is hint:
        return value
    raise ValueError("CCA facet value does not match its declared type")


def to_mlir(cca: CCA) -> str:
    """Emit complete analyzer facets with backward-compatible presentation views."""
    from xdsl.dialects.builtin import DictionaryAttr, ModuleOp, StringAttr
    from xdsl.ir import Block, Region

    from merlin.compiler.dialects import cca as D
    from merlin.compiler.dialects._common import text

    def _props(d: dict[str, str]) -> dict:
        return {k: StringAttr(v) for k, v in d.items()}

    inner = [D.ComputeOp(properties=_props(_compute_props(cca.compute)))]
    if cca.vector is not None:
        inner.append(D.VectorOp(properties=_props(_vector_props(cca.vector))))
    if cca.memory is not None:
        inner.append(D.MemoryOp(properties=_props(_memory_props(cca.memory))))
    # Legacy typed views remain inspectable; complete facets prevent their
    # older field subset from erasing dispatch, layout, communication or scope.
    for name, cls in _facet_classes().items():
        value = getattr(cca, name)
        if value is None:
            continue
        if not isinstance(value, cls):
            raise ValueError(f"CCA facet {name} has the wrong analyzer type")
        hints = typing.get_type_hints(cls)
        fields = {f.name: _typed_value(getattr(value, f.name), hints[f.name]) for f in dataclasses.fields(cls)}
        inner.append(
            D.FacetOp(
                properties={
                    "facet": StringAttr(name),
                    "values": DictionaryAttr({k: StringAttr(_json(v)) for k, v in fields.items()}),
                }
            )
        )
    kprops = {
        "op": cca.op,
        "backend": ",".join(cca.backend),
        "source": cca.provenance.get("source"),
        "level": cca.provenance.get("level"),
        "scope": cca.scope,
        "provenance": _json(cca.provenance),
        "backend_values": _json(cca.backend),
    }
    kernel = D.KernelOp(properties=_props({k: v for k, v in kprops.items() if v}), regions=[Region([Block(inner)])])
    return text(ModuleOp([kernel]))


def _get(op, field: str) -> str | None:
    a = getattr(op, field, None)
    return a.data if a is not None else None


def from_mlir(mlir_text: str) -> CCA:
    """Restore complete analyzer facets, retaining legacy subset artifact reads."""
    from xdsl.parser import Parser

    from merlin.compiler.dialects import cca as D
    from merlin.compiler.dialects._common import make_context

    module = Parser(make_context(D.get_dialect()), mlir_text).parse_module()
    module.verify()
    kernels = [o for o in module.body.block.ops if isinstance(o, D.KernelOp)]
    if len(kernels) != 1:
        raise ValueError("CCA artifact must contain exactly one kernel")
    kernel = kernels[0]
    facets = [o for o in kernel.body.block.ops if isinstance(o, D.FacetOp)]
    if facets:
        return _from_complete_facets(kernel, facets, D)
    compute_op = next((o for o in kernel.body.block.ops if isinstance(o, D.ComputeOp)), None)
    vector_op = next((o for o in kernel.body.block.ops if isinstance(o, D.VectorOp)), None)
    memory_op = next((o for o in kernel.body.block.ops if isinstance(o, D.MemoryOp)), None)

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
            activation_vectorization=_get(compute_op, "activation_vectorization"),
        )

    vector = None
    if vector_op is not None:
        sew = _get(vector_op, "sew")
        lmul = _get(vector_op, "lmul")
        vector = VectorFacet(
            sew=int(sew) if sew is not None else None,
            lmul=float(lmul) if lmul is not None else None,
            vl_strategy=_get(vector_op, "vl_strategy"),
            tail=_get(vector_op, "tail"),
        )

    memory = None
    if memory_op is not None:
        memory = MemoryFacet(
            access_pattern=_get(memory_op, "access_pattern"),
            panel_reuse=_b(_get(memory_op, "panel_reuse")),
            a_broadcast_vf=_b(_get(memory_op, "a_broadcast_vf")),
        )

    backend = _get(kernel, "backend")
    return CCA(
        op=_get(kernel, "op") or (compute.op or "unknown"),
        backend=backend.split(",") if backend else [],
        compute=compute,
        vector=vector,
        memory=memory,
        provenance={
            k: v for k, v in (("source", _get(kernel, "source")), ("level", _get(kernel, "level"))) if v is not None
        },
    )


def _from_complete_facets(kernel, facets, D) -> CCA:
    from xdsl.dialects.builtin import StringAttr

    classes = _facet_classes()
    restored = {}
    for op in facets:
        name = op.facet.data
        if name not in classes or name in restored:
            raise ValueError(f"unknown or duplicate CCA facet {name!r}")
        cls = classes[name]
        hints = typing.get_type_hints(cls)
        fields = {f.name for f in dataclasses.fields(cls)}
        if set(op.values.data) != fields:
            raise ValueError(f"CCA facet {name!r} does not contain its complete field set")
        values = {}
        for key, attr in op.values.data.items():
            if not isinstance(attr, StringAttr):
                raise ValueError("CCA facet values must contain JSON StringAttrs")
            values[key] = _typed_value(json.loads(attr.data), hints[key])
        restored[name] = cls(**values)
    if "compute" not in restored:
        raise ValueError("complete CCA artifact is missing its compute facet")
    backend = json.loads(_get(kernel, "backend_values") or "null")
    provenance = json.loads(_get(kernel, "provenance") or "null")
    scope = _get(kernel, "scope")
    if (
        not isinstance(backend, list)
        or any(type(v) is not str for v in backend)
        or not isinstance(provenance, dict)
        or scope is None
    ):
        raise ValueError("complete CCA artifact has malformed identity or provenance")
    if _get(kernel, "backend") != (",".join(backend) or None):
        raise ValueError("CCA backend view disagrees with its complete identity")
    for field in ("source", "level"):
        if _get(kernel, field) != (provenance.get(field) or None):
            raise ValueError(f"CCA {field} view disagrees with its complete provenance")
    # A caller editing the legacy presentation cannot silently disagree with
    # the complete analyzer record used by the comparator and route catalog.
    for name, cls, props in (
        ("compute", D.ComputeOp, _compute_props),
        ("vector", D.VectorOp, _vector_props),
        ("memory", D.MemoryOp, _memory_props),
    ):
        views = [op for op in kernel.body.block.ops if isinstance(op, cls)]
        expected = props(restored[name]) if name in restored else None
        if len(views) != (1 if expected is not None else 0):
            raise ValueError(f"CCA {name} view count disagrees with complete facets")
        if views and {k: v.data for k, v in views[0].properties.items()} != expected:
            raise ValueError(f"CCA {name} view disagrees with complete facets")
    return CCA(op=_get(kernel, "op") or "unknown", backend=backend, scope=scope, provenance=provenance, **restored)
