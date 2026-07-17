"""Re-emit a model SECTION as its own standalone MLIR ``@forward`` (route (a) for a section binary).

The C8 dispatch-DAG slicer (:func:`.dispatch_program.slice_program`) selects a section at the flat
runtime-program level. To build a *board binary* for that section we instead slice the OUTLINED MLIR:
keep the dispatch calls whose kernel symbol carries a selected ``prov.region_id`` (plus the interior
view-glue between them), promote the section's boundary inputs to function arguments, and return the
section's outputs. The result is an ordinary single-``@forward`` module holding only the kept kernel
funcs — so it flows through the EXISTING whole-model K1 build (``rvvgen.k1.build_k1_binary`` →
``lower_model_file`` → ``c_runtime.generate`` → ELF) unchanged, fed the region-boundary tensors as
inputs. Compile the whole model once; emit + run just the section you care about.

This is the MLIR analogue of ``slice_program`` (same keep-set logic, on SSA use-def instead of the
flat buffer table); the two agree on which kernels a section contains.
"""
from __future__ import annotations

from .._common import HAS_XDSL
from .outline import OutlineError, region_id_of_symbol


def _callee_symbol(call_op) -> str | None:
    """The callee symbol string of a ``func.call`` (tolerating xDSL API spellings)."""
    cal = getattr(call_op, "callee", None)
    if cal is None:
        return None
    for attr in ("string_value", "root_reference"):
        v = getattr(cal, attr, None)
        if callable(v):
            return v()
        if v is not None:
            return getattr(v, "data", str(v))
    return str(cal)


def emit_section_module(outlined_module, region_ids, *, entry: str = "forward"):
    """Build a standalone module whose ``@<entry>`` runs ONLY the section (the dispatch nodes whose
    kernel symbol carries a selected ``prov.region_id``) over its boundary inputs. Returns
    ``(module, boundary_inputs, section_outputs)`` where the latter two are the ordered SSA values
    (in the ORIGINAL module) that became the new function's args / results — the caller maps boundary
    inputs to region-boundary tensors and section outputs to the region goldens. Raises if nothing
    matches or the driver is not found."""
    if not HAS_XDSL:
        raise OutlineError("xDSL is required to emit a section module")
    from xdsl.dialects.builtin import FunctionType, ModuleOp, StringAttr
    from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    want = set(region_ids)
    funcs = [op for op in outlined_module.body.block.ops if isinstance(op, FuncOp)]
    driver = next((f for f in funcs if f.sym_name.data == entry), None)
    if driver is None:
        raise OutlineError(f"no driver func @{entry} in the outlined module")
    kernel_funcs = {f.sym_name.data: f for f in funcs if "$kernel_" in f.sym_name.data}
    block = driver.body.blocks[0]
    ops = list(block.ops)

    keep: set[int] = set()
    for i, op in enumerate(ops):
        if isinstance(op, CallOp):
            sym = _callee_symbol(op)
            if sym and region_id_of_symbol(sym) in want:
                keep.add(i)
    if not keep:
        raise OutlineError(f"emit_section_module: no dispatch matches region_ids {sorted(want)}")

    # Pull in interior view-glue: a non-call op is kept when EVERY use of its results is already kept
    # (it lives entirely inside the section). Fixpoint over SSA uses (mirrors slice_program).
    idx_of = {id(op): i for i, op in enumerate(ops)}
    changed = True
    while changed:
        changed = False
        for i, op in enumerate(ops):
            if i in keep or isinstance(op, (CallOp, ReturnOp)):
                continue
            uses = [u.operation for r in op.results for u in r.uses]
            if uses and all(idx_of.get(id(u)) in keep for u in uses):
                keep.add(i)
                changed = True

    kept_ops = [ops[i] for i in sorted(keep)]
    produced = {id(r) for op in kept_ops for r in op.results}

    # Boundary inputs: operands of kept ops not produced inside the section (block args or upstream
    # results) -> the section's function arguments, in first-use order.
    boundary: list = []
    seen_b: set[int] = set()
    for op in kept_ops:
        for operand in op.operands:
            if id(operand) not in produced and id(operand) not in seen_b:
                seen_b.add(id(operand))
                boundary.append(operand)

    # Section outputs: kept results used OUTSIDE the section or returned by the whole model.
    ret = next(o for o in ops if isinstance(o, ReturnOp))
    ret_vals = {id(o) for o in ret.operands}
    outputs: list = []
    seen_o: set[int] = set()
    for op in kept_ops:
        for r in op.results:
            used_outside = id(r) in ret_vals or any(idx_of.get(id(u.operation)) not in keep
                                                    for u in r.uses)
            if used_outside and id(r) not in seen_o:
                seen_o.add(id(r))
                outputs.append(r)

    # Build @entry(boundary...) -> (outputs...) cloning the kept ops with a value remap.
    new_block = Block(arg_types=[v.type for v in boundary])
    vmap = {b: arg for b, arg in zip(boundary, new_block.args)}
    for op in kept_ops:
        clone = op.clone(value_mapper=vmap)
        new_block.add_op(clone)
        for old, new in zip(op.results, clone.results):
            vmap[old] = new
    new_block.add_op(ReturnOp(*[vmap[o] for o in outputs]))
    section_fn = FuncOp(entry, FunctionType.from_lists([v.type for v in boundary],
                                                       [o.type for o in outputs]),
                        Region([new_block]))

    # Carry only the kernel funcs the section actually calls (clone so they detach cleanly).
    called = {_callee_symbol(op) for op in kept_ops if isinstance(op, CallOp)}
    kept_kernels = [kernel_funcs[s].clone() for s in called if s in kernel_funcs]
    module = ModuleOp([section_fn, *kept_kernels])
    module.verify()
    return module, boundary, outputs
