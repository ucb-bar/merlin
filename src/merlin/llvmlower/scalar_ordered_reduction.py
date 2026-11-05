"""Scalar SSA for closed one-input tensor reductions in their original order.

Retain the source seed, input mapping, binary32 operation/operand order and
increasing reduction index. Typed producer proofs exclude NaNs; addition also
requires finite inputs or a nonnegative input/seed domain. Unknown domains keep
their original reduction, preserving backend-specific NaN payload behavior.
Tensor SSA and ordinary bufferization continue to
own destination aliasing; this feature supplies no pointer or reassociation fact.
"""

from __future__ import annotations

from .ordered_reduction_domains import SOURCE as DOMAIN_SOURCE

FEATURE = "scalar_ordered_tensor_reductions"
MARKER = "__merlin_scalar_ordered_tensor_reductions__"


def edit_pipeline(passes):
    positions = [i for i, stage in enumerate(passes) if "one-shot-bufferize" in stage]
    if len(positions) != 1 or MARKER in passes:
        raise ValueError("ordered tensor reductions require one bufferization boundary")
    at = min([positions[0], *[i for i, stage in enumerate(passes) if "linalg-fuse-elementwise" in stage]])
    return [*passes[:at], MARKER, *passes[at:]]


def ensure_registered():
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description=(
                    "Keep proved non-NaN binary32 add/max reductions in ordered scalar SSA, "
                    "preserving seed, operand order and tensor alias authority. Default off."
                ),
                edit_pipeline=edit_pipeline,
            )
        )
    return FEATURE


REWRITE_SOURCE = (
    DOMAIN_SOURCE
    + r"""
def _ordered_reduction_spec(op):
    from torch_mlir import ir as _or_ir

    if (
        op.operation.name not in {"linalg.generic", "linalg.reduce"}
        or len(op.operands) != 2
        or len(op.results) != 1
        or len(op.regions) != 1
        or len(op.regions[0].blocks) != 1
    ):
        return None
    ancestor = op.operation
    while ancestor is not None:
        if any(
            "strictfp" in name or "strictfp" in str(ancestor.attributes[name])
            for name in ancestor.attributes
        ):
            return None
        ancestor = ancestor.parent
    types = []
    for value in op.operands:
        if not isinstance(value.type, _or_ir.RankedTensorType):
            return None
        tensor = _or_ir.RankedTensorType(value.type)
        if not isinstance(tensor.element_type, _or_ir.F32Type) or any(
            extent < 0 for extent in tensor.shape
        ):
            return None
        types.append(tensor)
    if op.results[0].type != op.operands[1].type:
        return None
    allowed = (
        {"dimensions"}
        if op.operation.name == "linalg.reduce"
        else {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    )
    if any(
        name not in allowed and not name.startswith("prov.") for name in op.attributes
    ):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (
        len(block.arguments) != 2
        or len(body) != 2
        or body[0].operation.name
        not in {"arith.addf", "arith.maximumf", "arith.maxnumf"}
        or body[1].operation.name != "linalg.yield"
        or any(x.regions for x in body)
        or any(
            name != "fastmath" and not name.startswith("prov.")
            for scalar in body
            for name in scalar.attributes
        )
        or any(
            "fastmath" in scalar.attributes
            and str(scalar.attributes["fastmath"]) != "#arith.fastmath<none>"
            for scalar in body
        )
    ):
        return None
    scalar, terminator = body
    arguments = list(block.arguments)
    if list(scalar.operands) == arguments:
        order = [0, 1]
    elif list(scalar.operands) == arguments[::-1]:
        order = [1, 0]
    else:
        return None
    if list(terminator.operands) != [scalar.results[0]]:
        return None
    if op.operation.name == "linalg.reduce":
        dims = len(types[0].shape)
        reduced = list(_or_ir.DenseI64ArrayAttr(op.attributes["dimensions"]))
        if (
            dims < 1
            or reduced != [dims - 1]
            or list(types[1].shape) != list(types[0].shape[:-1])
        ):
            return None
        positions, extents = (
            [list(range(dims)), list(range(dims - 1))],
            list(types[0].shape),
        )
    else:
        maps = [_or_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
        if len(maps) != 2:
            return None
        dims = maps[0].n_dims
        if (
            dims < 1
            or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]]
            != ["#linalg.iterator_type<parallel>"] * (dims - 1)
            + ["#linalg.iterator_type<reduction>"]
        ):
            return None
        extents, positions = [None] * dims, []
        for affine, tensor in zip(maps, types):
            if len(affine.results) != len(tensor.shape):
                return None
            used = []
            for expr, extent in zip(affine.results, tensor.shape):
                if not isinstance(expr, _or_ir.AffineDimExpr):
                    return None
                position = _or_ir.AffineDimExpr(expr).position
                if (
                    position in used
                    or position >= dims
                    or (extents[position] is not None and extents[position] != extent)
                ):
                    return None
                used.append(position)
                extents[position] = extent
            positions.append(used)
        if (
            sorted(positions[0]) != list(range(dims))
            or sorted(positions[1]) != list(range(dims - 1))
            or any(extent is None for extent in extents)
        ):
            return None
    proof = _OrderedReductionDomains()
    a, b = [proof.tensor(value) for value in op.operands]
    if a is None or b is None:
        return None
    if scalar.operation.name == "arith.addf" and not (
        (a[0] and b[0]) or (a[1] and b[1])
    ):
        return None
    return positions, extents, order


def _scalarize_ordered_reductions(ctx, module):
    from torch_mlir import ir as _or_ir

    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    spec = _ordered_reduction_spec(child)
                    if spec is not None:
                        todo.append((child, spec))
                    else:
                        walk(child.operation)

    walk(module.operation)
    with ctx:
        for old, (positions, extents, operand_order) in todo:
            with old.location, _or_ir.InsertionPoint(old):
                index = _or_ir.IndexType.get()
                scalar = old.regions[0].blocks[0].arguments[0].type
                tensor = old.results[0].type

                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _or_ir.Operation.create(
                        name,
                        operands=list(operands),
                        results=list(results),
                        attributes=attributes or {},
                        regions=regions,
                    )

                def constant(value):
                    return create(
                        "arith.constant",
                        results=[index],
                        attributes={"value": _or_ir.IntegerAttr.get(index, value)},
                    ).results[0]

                zero, one = constant(0), constant(1)
                limits = [constant(extent) for extent in extents]
                source = list(old.regions[0].blocks[0].operations)[0]

                def output_loop(depth, current, coordinates):
                    if depth < len(extents) - 1:
                        loop = create(
                            "scf.for",
                            [zero, limits[depth], one, current],
                            [tensor],
                            regions=1,
                        )
                        block = _or_ir.Block.create_at_start(
                            loop.regions[0], [index, tensor]
                        )
                        with _or_ir.InsertionPoint(block):
                            updated = output_loop(
                                depth + 1,
                                block.arguments[1],
                                [*coordinates, block.arguments[0]],
                            )
                            create("scf.yield", [updated])
                        return loop.results[0]
                    indices = [coordinates[d] for d in positions[1]]
                    initial = create(
                        "tensor.extract", [current, *indices], [scalar]
                    ).results[0]
                    loop = create(
                        "scf.for", [zero, limits[-1], one, initial], [scalar], regions=1
                    )
                    block = _or_ir.Block.create_at_start(
                        loop.regions[0], [index, scalar]
                    )
                    with _or_ir.InsertionPoint(block):
                        all_indices = [*coordinates, block.arguments[0]]
                        value = create(
                            "tensor.extract",
                            [old.operands[0], *[all_indices[d] for d in positions[0]]],
                            [scalar],
                        ).results[0]
                        values = [value, block.arguments[1]]
                        result = create(
                            source.operation.name,
                            [values[d] for d in operand_order],
                            [scalar],
                            {
                                name: source.attributes[name]
                                for name in source.attributes
                            },
                        ).results[0]
                        create("scf.yield", [result])
                    return create(
                        "tensor.insert", [loop.results[0], current, *indices], [tensor]
                    ).results[0]

                result = output_loop(0, old.operands[1], [])
                for name in old.attributes:
                    if name.startswith("prov."):
                        result.owner.attributes[name] = old.attributes[name]
                previous = (
                    old.attributes["prov.transforms"]
                    if "prov.transforms" in old.attributes
                    else None
                )
                prefix = (
                    _or_ir.StringAttr(previous).value + ","
                    if previous is not None
                    else ""
                )
                result.owner.attributes["prov.transforms"] = _or_ir.StringAttr.get(
                    prefix + "scalar_ordered_tensor_reductions"
                )
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)
"""
)

RUNNER_PRELUDE = (
    REWRITE_SOURCE
    + r"""
_ORDERED_REDUCTION_ORIGINAL = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    marker = '__merlin_scalar_ordered_tensor_reductions__'
    if marker not in passes:
        return _ORDERED_REDUCTION_ORIGINAL(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(marker) != 1:
        raise ValueError('ordered reduction marker must occur exactly once')
    at = passes.index(marker)
    head_has_generalize = any('linalg-generalize-named-ops' in stage for stage in passes[:at])
    _ORDERED_REDUCTION_ORIGINAL(ctx, module, ','.join(passes[:at]), 0, (), (), (),
                                pre_generalize if head_has_generalize else ())
    print('OK scalar_ordered_tensor_reductions', _scalarize_ordered_reductions(ctx, module))
    return _ORDERED_REDUCTION_ORIGINAL(ctx, module, ','.join(passes[at + 1:]), erase, mid, late, post_openmp,
                                     () if head_has_generalize else pre_generalize)
"""
)


def apply_for_test(mlir_text, *, directory):
    """Run the shipped rewrite in a fresh native MLIR context and caller workdir."""
    import subprocess
    from pathlib import Path

    from .toolchain import m2m_python

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    source, script = root / "input.mlir", root / "run.py"
    source.write_text(mlir_text)
    script.write_text(
        "import sys\nfrom torch_mlir import ir\n"
        + REWRITE_SOURCE
        + "\nctx=ir.Context()\nmodule=ir.Module.parse(open(sys.argv[1]).read(),ctx)\n"
        + "print('COUNT',_scalarize_ordered_reductions(ctx,module))\nprint('MODULE_BEGIN')\nprint(module)\n"
    )
    result = subprocess.run([str(m2m_python()), str(script), str(source)], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"ordered reduction rewrite failed:\n{result.stdout}\n{result.stderr}")
    count = int(next(line.split()[1] for line in result.stdout.splitlines() if line.startswith("COUNT ")))
    return result.stdout.split("MODULE_BEGIN\n", 1)[1], count
