"""Explicit source-semantic hoisting of broadcast reciprocal square roots.

Run after elementwise fusion and before bufferization. Exact affine projections
and scalar SSA dependencies establish a smaller, nonempty producer domain. No
external-call name grants purity, no arithmetic is reassociated, and tensor
value semantics leave physical alias/lifetime decisions to bufferization.
"""

from __future__ import annotations

FEATURE = "hoist_broadcast_source_rsqrt"
MARKER = "__merlin_broadcast_source_rsqrt_hoist__"


def _edit_pipeline(passes: list[str]) -> list[str]:
    hits = [i for i, stage in enumerate(passes) if "one-shot-bufferize" in stage]
    if len(hits) != 1 or MARKER in passes:
        raise ValueError("broadcast math hoist requires one bufferization stage")
    at = hits[0]
    return [*passes[:at], MARKER, *passes[at:]]


def ensure_registered() -> str:
    from .shared_feature_registry import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description=(
                    "Hoist typed pure source rsqrt dependency chains onto their exact "
                    "static nonempty broadcast domain after fusion; preserve every "
                    "operation, type and other SSA use. No strict FP/fenv permission."
                ),
                edit_pipeline=_edit_pipeline,
            )
        )
    return FEATURE


RUNNER_PRELUDE = r"""
def _broadcast_math_spec(op):
    from torch_mlir import ir as _bh_ir
    if any("strictfp" in name or "strictfp" in str(op.attributes[name])
           for name in op.attributes):
        return None
    if (op.operation.name != "linalg.generic" or len(op.results) != 1
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
            or len(op.operands) < 2):
        return None
    types = [value.type for value in op.operands]
    if any(not isinstance(t, _bh_ir.RankedTensorType)
           or any(extent < 0 for extent in t.shape) for t in types):
        return None
    maps = [_bh_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    dims = maps[-1].n_dims
    if (dims < 1 or len(maps) != len(types)
            or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]]
               != ["#linalg.iterator_type<parallel>"] * dims):
        return None
    # Every domain dimension is fixed by the complete output permutation.
    out_dims = []
    extents = [None] * dims
    for expr, extent in zip(maps[-1].results, types[-1].shape):
        if not isinstance(expr, _bh_ir.AffineDimExpr):
            return None
        dimension = _bh_ir.AffineDimExpr(expr).position
        if dimension in out_dims:
            return None
        out_dims.append(dimension)
        extents[dimension] = extent
    if (sorted(out_dims) != list(range(dims))
            or any(extent is None or extent <= 0 for extent in extents)):
        # A producer must never evaluate an operation absent from an empty consumer.
        return None
    dependencies = []
    for affine, tensor in zip(maps, types):
        if len(affine.results) != len(tensor.shape):
            return None
        used = set()
        for expr, extent in zip(affine.results, tensor.shape):
            if isinstance(expr, _bh_ir.AffineDimExpr):
                dimension = _bh_ir.AffineDimExpr(expr).position
                if dimension in used or extents[dimension] != extent:
                    return None
                used.add(dimension)
            elif isinstance(expr, _bh_ir.AffineConstantExpr):
                if _bh_ir.AffineConstantExpr(expr).value != 0 or extent != 1:
                    return None
            else:
                return None
        dependencies.append(frozenset(used))
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != len(types) or not body
            or body[-1].operation.name != "linalg.yield"
            or len(body[-1].operands) != 1):
        return None
    # These registered source operations are scalar and have no observable effects.
    # Reject an entire body containing a call/index/memory/unknown operation: moving
    # a pure chain across it could change an observed floating environment.
    allowed = {"arith.constant", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
               "arith.negf", "arith.maximumf", "arith.minimumf", "arith.cmpf", "arith.cmpi",
               "arith.select", "arith.andi", "arith.ori", "arith.xori", "arith.addi", "arith.subi",
               "arith.muli", "arith.shli", "arith.shrsi", "arith.shrui", "arith.bitcast",
               "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui", "arith.extsi",
               "arith.extui", "arith.trunci", "arith.extf", "arith.truncf", "math.rsqrt"}
    scalar_deps = dict(zip(block.arguments, dependencies))
    defining = {}
    candidates = []
    for scalar in body[:-1]:
        if any("strictfp" in name or "strictfp" in str(scalar.attributes[name])
               for name in scalar.attributes):
            return None
        if (scalar.operation.name not in allowed or scalar.regions
                or len(scalar.results) != 1
                or isinstance(scalar.results[0].type, (_bh_ir.VectorType, _bh_ir.ShapedType))):
            return None
        if ("fastmath" in scalar.attributes
                and str(scalar.attributes["fastmath"]) != "#arith.fastmath<none>"):
            return None
        deps = frozenset().union(*(scalar_deps.get(value, frozenset())
                                  for value in scalar.operands))
        scalar_deps[scalar.results[0]] = deps
        defining[scalar.results[0]] = scalar
        if (scalar.operation.name == "math.rsqrt"
                and isinstance(scalar.results[0].type,
                               (_bh_ir.F16Type, _bh_ir.F32Type, _bh_ir.F64Type))
                and any(d not in deps and extents[d] > 1 for d in range(dims))):
            candidates.append(scalar)
    if not candidates:
        return None
    target = candidates[0]
    needed = set()
    def need(value):
        if value in needed:
            return
        needed.add(value)
        scalar = defining.get(value)
        if scalar is not None:
            for operand in scalar.operands:
                need(operand)
    need(target.results[0])
    # A dependency on an output-init block argument cannot be read by a fresh producer.
    if block.arguments[-1] in needed:
        return None
    inputs = [i for i, value in enumerate(block.arguments[:-1]) if value in needed]
    chain = [scalar for scalar in body[:-1] if scalar.results[0] in needed]
    keep = sorted(scalar_deps[target.results[0]])
    return maps, extents, body, target, inputs, chain, keep


def _hoist_broadcast_source_math(ctx, module):
    from torch_mlir import ir as _bh_ir
    todo = []
    def walk(operation):
        if any("strictfp" in name or "strictfp" in str(operation.attributes[name])
               for name in operation.attributes):
            return
        for region in operation.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _broadcast_math_spec(inner)
                    if spec is not None:
                        todo.append((inner, spec))
                    else:
                        walk(inner.operation)
    walk(module.operation)
    with ctx:
        for old, (maps, extents, body, target, inputs, chain, keep) in todo:
            old_args = list(old.regions[0].blocks[0].arguments)
            with old.location:
                reduced_type = _bh_ir.RankedTensorType.get(
                    [extents[d] for d in keep], target.results[0].type)
            def create(name, operands=(), results=(), attributes=None, regions=0):
                return _bh_ir.Operation.create(name, operands=list(operands), results=list(results),
                                              attributes=attributes or {}, regions=regions)
            def attrs(operation):
                return {name: operation.attributes[name] for name in operation.attributes}
            def clone(scalar, values):
                with scalar.location:
                    copy = create(scalar.operation.name,
                                  [values.get(v, v) for v in scalar.operands],
                                  [v.type for v in scalar.results], attrs(scalar))
                values[scalar.results[0]] = copy.results[0]
            def map_on_reduced(affine):
                results = []
                for expr in affine.results:
                    if isinstance(expr, _bh_ir.AffineDimExpr):
                        position = _bh_ir.AffineDimExpr(expr).position
                        results.append(_bh_ir.AffineDimExpr.get(keep.index(position)))
                    else:
                        results.append(_bh_ir.AffineConstantExpr.get(0))
                return _bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get(len(keep), 0, results))
            with old.location, _bh_ir.InsertionPoint(old):
                empty = create("tensor.empty", results=[reduced_type])
                producer_attrs = {
                    "indexing_maps": _bh_ir.ArrayAttr.get(
                        [map_on_reduced(maps[i]) for i in inputs]
                        + [_bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get_identity(len(keep)))]),
                    "iterator_types": _bh_ir.ArrayAttr.get(
                        [_bh_ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(keep)),
                    "operandSegmentSizes": _bh_ir.DenseI32ArrayAttr.get([len(inputs), 1]),
                }
                producer = create("linalg.generic", [*[old.operands[i] for i in inputs], empty.results[0]],
                                  [reduced_type], producer_attrs, 1)
                producer_block = _bh_ir.Block.create_at_start(
                    producer.regions[0], [*[old_args[i].type for i in inputs], target.results[0].type])
                with _bh_ir.InsertionPoint(producer_block):
                    values = {old_args[i]: producer_block.arguments[j] for j, i in enumerate(inputs)}
                    for scalar in chain:
                        clone(scalar, values)
                    create("linalg.yield", [values[target.results[0]]])
                consumer_attrs = attrs(old)
                consumer_attrs["indexing_maps"] = _bh_ir.ArrayAttr.get(
                    [*[_bh_ir.AffineMapAttr.get(m) for m in maps[:-1]],
                     _bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get(maps[-1].n_dims, 0,
                                               [_bh_ir.AffineDimExpr.get(d) for d in keep])),
                     _bh_ir.AffineMapAttr.get(maps[-1])])
                consumer_attrs["operandSegmentSizes"] = _bh_ir.DenseI32ArrayAttr.get([len(old.operands), 1])
                consumer = create("linalg.generic", [*old.operands[:-1], producer.results[0], old.operands[-1]],
                                  [old.results[0].type], consumer_attrs, 1)
                consumer_block = _bh_ir.Block.create_at_start(
                    consumer.regions[0], [*[v.type for v in old_args[:-1]],
                                         target.results[0].type, old_args[-1].type])
                with _bh_ir.InsertionPoint(consumer_block):
                    values = {old_args[i]: consumer_block.arguments[i] for i in range(len(old_args) - 1)}
                    values[old_args[-1]] = consumer_block.arguments[-1]
                    values[target.results[0]] = consumer_block.arguments[-2]
                    # Backward liveness preserves every other use of a chain intermediate.
                    live = set(body[-1].operands)
                    retained = []
                    for scalar in reversed(body[:-1]):
                        if scalar is target:
                            continue
                        if scalar.results[0] in live:
                            retained.append(scalar)
                            live.update(scalar.operands)
                    for scalar in reversed(retained):
                        clone(scalar, values)
                    create("linalg.yield", [values.get(v, v) for v in body[-1].operands])
            old.results[0].replace_all_uses_with(consumer.results[0])
            old.operation.erase()
    module.operation.verify()
    return len(todo)

_BH_MARKER = "__merlin_broadcast_source_rsqrt_hoist__"
_BH_ORIG_RUN_STAGES = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _BH_MARKER not in passes:
        return _BH_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(_BH_MARKER) != 1:
        raise ValueError("broadcast source math hoist marker must occur once")
    at = passes.index(_BH_MARKER)
    _BH_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:at]), 0, (), (), (), pre_generalize)
    print('OK broadcast_source_rsqrt_hoist', _hoist_broadcast_source_math(ctx, module))
    _BH_ORIG_RUN_STAGES(ctx, module, ','.join(passes[at + 1:]), erase, mid, late, post_openmp, ())
"""


def apply_for_test(mlir_text: str) -> tuple[str, int]:
    """Execute the shipped typed rewrite through the selected upstream compiler."""
    import subprocess
    import tempfile
    from pathlib import Path

    from .toolchain import m2m_python

    with tempfile.TemporaryDirectory(prefix="merlin_broadcast_math_") as directory:
        root = Path(directory)
        source, script = root / "input.mlir", root / "run.py"
        source.write_text(mlir_text)
        script.write_text(
            "import sys\nfrom torch_mlir import ir\n"
            + RUNNER_PRELUDE.split("_BH_MARKER =", 1)[0]
            + "ctx=ir.Context()\nmodule=ir.Module.parse(open(sys.argv[1]).read(),ctx)\n"
            + "print('COUNT',_hoist_broadcast_source_math(ctx,module))\nprint('MODULE_BEGIN')\nprint(module)\n"
        )
        result = subprocess.run(
            [str(m2m_python()), str(script), str(source)], capture_output=True, text=True, timeout=120
        )
        if result.returncode:
            raise RuntimeError(f"broadcast source math hoist failed:\n{result.stdout}\n{result.stderr}")
        count = int(next(line.split()[1] for line in result.stdout.splitlines() if line.startswith("COUNT ")))
        return result.stdout.split("MODULE_BEGIN\n", 1)[1], count
