"""Extract one real matmul SECTION from a model2MLIR linalg module as a standalone ``@section``.

Increment 3 (:mod:`.compilability`) reports what a real whole model contains; this pass carves out a
single, fully-modeled slice — one ``linalg.matmul`` and its int8 weight-only dequant prep — and
re-emits it as an ordinary linalg function that flows straight through ``lower_module`` and runs on
the engine. It is the linalg analogue of :func:`.section_mlir.emit_section_module` (which slices an
already-outlined dispatch module); here we slice raw linalg by SSA use-def.

The section keeps the matmul, the ``quant_ext.dequantize_per_channel`` feeding its RHS, and the
constant-like producers of those ops (the zero-point splat, the accumulator init) so they stay
internal. The section's boundary inputs — in first-use order — are the activation (matmul LHS), the
i8 weight (the dequant's quantized input, i.e. the transposed weight), and the per-channel scale; its
single result is the matmul output. The caller injects those three and grades the result.
"""
from __future__ import annotations

from .._common import HAS_XDSL
from .interface_lowering import _resolved_name

_CONSTANT_LIKE = ("arith.constant", "tensor.splat", "tensor.empty")
_VIEW_OPS = ("tensor.expand_shape", "tensor.collapse_shape", "tensor.cast", "linalg.copy",
             "linalg.transpose")


def _prov_region(op) -> str | None:
    from xdsl.dialects.builtin import StringAttr

    val = op.attributes.get("prov.region_id")
    return val.data if isinstance(val, StringAttr) else None


def _find_matmul(module, region_id):
    for op in module.walk():
        if op.name == "linalg.matmul" and (region_id is None or _prov_region(op) == region_id):
            return op
    return None


def _dequant_feeding(rhs):
    """The ``quant_ext.dequantize_per_channel`` op feeding a matmul RHS (through any views), or None."""
    cur = rhs
    for _ in range(8):
        owner = getattr(cur, "owner", None)
        nm = _resolved_name(owner)
        if nm in _VIEW_OPS:
            cur = owner.inputs[0] if nm == "linalg.transpose" else owner.operands[0]
            continue
        if nm is not None and nm.startswith("quant_ext.dequantize"):
            return owner
        return None
    return None


def section_from_matmul(module, region_id=None):
    """Build ``(section_module, boundary_values, [matmul_result])`` for the matmul with the given
    ``prov.region_id`` (or the first matmul when None). Raises if no matmul or no dequant is found."""
    if not HAS_XDSL:
        raise RuntimeError("xDSL is required for section extraction")
    from xdsl.dialects.builtin import FunctionType, ModuleOp
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    mm = _find_matmul(module, region_id)
    if mm is None:
        raise ValueError(f"no linalg.matmul with region_id {region_id!r}")
    dq = _dequant_feeding(mm.inputs[1])
    if dq is None:
        raise ValueError("matmul RHS is not fed by a quant_ext.dequantize (int8 weight-only) op")

    # Seed the keep-set with the matmul + its dequant, then pull in the constant-like producers of
    # any kept operand (zero-point splat, init tensors) so only genuine boundary values stay outside.
    keep_ids: set[int] = {id(mm), id(dq)}
    seeds = [mm, dq]
    i = 0
    while i < len(seeds):
        for operand in seeds[i].operands:
            producer = getattr(operand, "owner", None)
            if (producer is not None and _resolved_name(producer) in _CONSTANT_LIKE
                    and id(producer) not in keep_ids):
                keep_ids.add(id(producer))
                seeds.append(producer)
        i += 1

    # Kept ops in program order (walk the whole module; keep-set membership decides).
    kept = [op for op in module.walk() if id(op) in keep_ids]
    produced = {id(r) for op in kept for r in op.results}

    boundary: list = []
    seen: set[int] = set()
    for op in kept:
        for operand in op.operands:
            if id(operand) not in produced and id(operand) not in seen:
                seen.add(id(operand))
                boundary.append(operand)

    new_block = Block(arg_types=[v.type for v in boundary])
    vmap = {b: arg for b, arg in zip(boundary, new_block.args)}
    for op in kept:
        clone = op.clone(value_mapper=vmap)
        new_block.add_op(clone)
        for old, new in zip(op.results, clone.results):
            vmap[old] = new
    result = vmap[mm.results[0]]
    new_block.add_op(ReturnOp(result))
    fn = FuncOp("section", FunctionType.from_lists([v.type for v in boundary], [result.type]),
                Region([new_block]))
    module_out = ModuleOp([fn])
    return module_out, boundary, [mm.results[0]]
