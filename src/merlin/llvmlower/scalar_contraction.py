"""Exact scalar accumulators for static tensor contractions, before bufferization.

The final reduction dimension runs in increasing order with the original separate
multiply and add. Its scalar result is an scf.for iter_arg; the destination tensor
is updated once per output element. Tensor value semantics leave input/destination
alias decisions to upstream bufferization rather than asserting physical noalias.
"""

from __future__ import annotations

FEATURE = "scalar_contraction_accumulator"
MARKER = "__merlin_scalar_contraction_accumulator__"
TWO_OUTPUTS_FEATURE = "scalar_contraction_accumulator_2_outputs"
TWO_OUTPUTS_MARKER = "__merlin_scalar_contraction_accumulator_2_outputs__"
FOUR_OUTPUTS_FEATURE = "scalar_contraction_accumulator_4_outputs"
FOUR_OUTPUTS_MARKER = "__merlin_scalar_contraction_accumulator_4_outputs__"
EIGHT_OUTPUTS_FEATURE = "scalar_contraction_accumulator_8_outputs"
EIGHT_OUTPUTS_MARKER = "__merlin_scalar_contraction_accumulator_8_outputs__"
RECTANGULAR_FEATURE = "scalar_contraction_accumulator_2x4_outputs"
RECTANGULAR_MARKER = "__merlin_scalar_contraction_accumulator_2x4_outputs__"
REDUCTION_UNROLL_FEATURE = "unroll_scalar_contraction_reduction_by_2"
REDUCTION_UNROLL_MARKER = "__merlin_scalar_contraction_reduction_unroll_2__"
FOUR_REDUCTION_UNROLL_FEATURE = "unroll_scalar_contraction_reduction_by_4"
FOUR_REDUCTION_UNROLL_MARKER = "__merlin_scalar_contraction_reduction_unroll_4__"
_OUTPUTS_GROUP = "scalar_contraction_accumulator_outputs"
_UNROLL_GROUP = "scalar_contraction_reduction_unroll"
_OUTPUTS_FEATURES = frozenset(
    {FEATURE, TWO_OUTPUTS_FEATURE, FOUR_OUTPUTS_FEATURE, EIGHT_OUTPUTS_FEATURE, RECTANGULAR_FEATURE}
)


def _edit_pipeline(passes: list[str], *, outputs: int = 1, rows: int = 1) -> list[str]:
    if any(
        m in passes for m in (MARKER, TWO_OUTPUTS_MARKER, FOUR_OUTPUTS_MARKER, EIGHT_OUTPUTS_MARKER, RECTANGULAR_MARKER)
    ):
        raise ValueError("scalar contraction accumulator schedules are alternatives")
    hits = [i for i, p in enumerate(passes) if "one-shot-bufferize" in p]
    if len(hits) != 1:
        raise ValueError(f"{FEATURE} requires exactly one one-shot-bufferize stage")
    fusion = [i for i, p in enumerate(passes[: hits[0]]) if "linalg-fuse-elementwise-ops" in p]
    i = fusion[0] if fusion else hits[0]
    marker = (
        RECTANGULAR_MARKER
        if rows == 2
        else {1: MARKER, 2: TWO_OUTPUTS_MARKER, 4: FOUR_OUTPUTS_MARKER, 8: EIGHT_OUTPUTS_MARKER}[outputs]
    )
    return [*passes[:i], marker, *passes[i:]]


def _edit_reduction_unroll(passes: list[str], *, count: int = 2) -> list[str]:
    if any(m in passes for m in (REDUCTION_UNROLL_MARKER, FOUR_REDUCTION_UNROLL_MARKER)):
        raise ValueError("scalar contraction reduction unroll counts are alternatives")
    hits = [
        i
        for i, p in enumerate(passes)
        if p in (MARKER, TWO_OUTPUTS_MARKER, FOUR_OUTPUTS_MARKER, EIGHT_OUTPUTS_MARKER, RECTANGULAR_MARKER)
    ]
    if len(hits) != 1:
        raise ValueError(f"{REDUCTION_UNROLL_FEATURE} requires one scalar accumulator schedule")
    i = hits[0]
    marker = {2: REDUCTION_UNROLL_MARKER, 4: FOUR_REDUCTION_UNROLL_MARKER}[count]
    return [*passes[:i], marker, *passes[i:]]


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description="Keep exact ordered f32 tensor contraction accumulators in scalar loop arguments before bufferization.",
                edit_pipeline=_edit_pipeline,
                alternative_group=_OUTPUTS_GROUP,
            )
        )
    if FOUR_OUTPUTS_FEATURE not in known():
        register(
            ImprFeature(
                name=FOUR_OUTPUTS_FEATURE,
                action_class="PASS",
                description="Carry four independent exact output accumulators across increasing-K tensor contractions whose last parallel extent is divisible by four.",
                edit_pipeline=lambda passes: _edit_pipeline(passes, outputs=4),
                alternative_group=_OUTPUTS_GROUP,
            )
        )
    if EIGHT_OUTPUTS_FEATURE not in known():
        register(
            ImprFeature(
                name=EIGHT_OUTPUTS_FEATURE,
                action_class="PASS",
                description="Carry eight independent exact output accumulators across increasing-K tensor contractions whose last parallel extent is divisible by eight.",
                edit_pipeline=lambda passes: _edit_pipeline(passes, outputs=8),
                alternative_group=_OUTPUTS_GROUP,
            )
        )
    if TWO_OUTPUTS_FEATURE not in known():
        register(
            ImprFeature(
                name=TWO_OUTPUTS_FEATURE,
                action_class="PASS",
                description="Carry two independent exact output accumulators across increasing-K tensor contractions whose last parallel extent is divisible by two.",
                edit_pipeline=lambda passes: _edit_pipeline(passes, outputs=2),
                alternative_group=_OUTPUTS_GROUP,
            )
        )
    if RECTANGULAR_FEATURE not in known():
        register(
            ImprFeature(
                name=RECTANGULAR_FEATURE,
                action_class="PASS",
                description="Carry eight exact ordered accumulators across a two-row/four-column tensor contraction tile, sharing only typed coordinate-identical immutable input reads. Partial tiles refuse.",
                edit_pipeline=lambda passes: _edit_pipeline(passes, outputs=4, rows=2),
                alternative_group=_OUTPUTS_GROUP,
            )
        )
    if REDUCTION_UNROLL_FEATURE not in known():
        register(
            ImprFeature(
                name=REDUCTION_UNROLL_FEATURE,
                action_class="PASS",
                description="Request partial unrolling by two for exact scalar contraction reduction loops using upstream LLVM loop annotations.",
                edit_pipeline=_edit_reduction_unroll,
                alternative_group=_UNROLL_GROUP,
                requires_exactly_one_of=_OUTPUTS_FEATURES,
            )
        )
    if FOUR_REDUCTION_UNROLL_FEATURE not in known():
        register(
            ImprFeature(
                name=FOUR_REDUCTION_UNROLL_FEATURE,
                action_class="PASS",
                description="Request partial unrolling by four for exact scalar contraction reduction loops using upstream LLVM loop annotations.",
                edit_pipeline=lambda passes: _edit_reduction_unroll(passes, count=4),
                alternative_group=_UNROLL_GROUP,
                requires_exactly_one_of=_OUTPUTS_FEATURES,
            )
        )
    return FEATURE


RUNNER_PRELUDE = r"""
def _scalar_contraction_spec(op):
    from torch_mlir import ir as _sc_ir
    if (op.operation.name != "linalg.generic" or len(op.operands) != 3
            or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        return None
    types = list(v.type for v in op.operands)
    if any(not isinstance(t, _sc_ir.RankedTensorType) for t in types):
        return None
    types = [_sc_ir.RankedTensorType(t) for t in types]
    if (any(not isinstance(t.element_type, _sc_ir.F32Type) for t in types)
            or op.results[0].type != op.operands[2].type
            or any(d < 0 for t in types for d in t.shape)):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != 3 or len(body) != 3
            or [x.operation.name for x in body] != ["arith.mulf", "arith.addf", "linalg.yield"]
            or any("fastmath" in x.attributes and str(x.attributes["fastmath"]) != "#arith.fastmath<none>" for x in body)
            or any(name != "fastmath" and not name.startswith("prov.")
                   for x in body[:2] for name in x.attributes)):
        return None
    mul, add, yield_op = body
    args = list(block.arguments)
    if list(mul.operands) == args[:2]:
        mul_order = [0, 1]
    elif list(mul.operands) == args[1::-1]:
        mul_order = [1, 0]
    else:
        return None
    if list(add.operands) == [mul.results[0], args[2]]:
        add_order = [0, 1]
    elif list(add.operands) == [args[2], mul.results[0]]:
        add_order = [1, 0]
    else:
        return None
    if list(yield_op.operands) != [add.results[0]]:
        return None
    maps = [_sc_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if len(maps) != 3:
        return None
    dims = maps[0].n_dims
    if dims < 2 or any(m.n_dims != dims or m.n_symbols for m in maps):
        return None
    iters = list(op.attributes["iterator_types"])
    # Enum attribute printer is the upstream binding's stable spelling.
    if ([str(x) for x in iters] !=
            ["#linalg.iterator_type<parallel>"] * (dims - 1) + ["#linalg.iterator_type<reduction>"]):
        return None
    positions = []
    extents = [None] * dims
    for m, t in zip(maps, types):
        if len(m.results) != len(t.shape):
            return None
        pos = []
        for expr, extent in zip(m.results, t.shape):
            if not isinstance(expr, _sc_ir.AffineDimExpr):
                return None
            d = _sc_ir.AffineDimExpr(expr).position
            if d in pos or (extents[d] is not None and extents[d] != extent):
                return None
            pos.append(d)
            extents[d] = extent
        positions.append(pos)
    if (sorted(positions[2]) != list(range(dims - 1))
            or any(dims - 1 not in pos for pos in positions[:2])
            or any(d is None for d in extents)):
        return None
    return positions, extents, mul_order, add_order


def _scalarize_tensor_contractions(ctx, module, outputs=1, reduction_unroll=0, rows=1, selection=None, output_guard=None):
    from torch_mlir import ir as _sc_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _scalar_contraction_spec(inner)
                    eligible = (spec is not None and spec[1][-2] % outputs == 0
                                and (selection is None or any(inner.operation == item for item in selection)))
                    if eligible and rows != 1:
                        positions, extents, _, _ = spec
                        row_axis, col_axis = len(extents) - 3, len(extents) - 2
                        # A rectangular tile is a matrix schedule only when the
                        # exact projected maps prove complementary row/column
                        # reuse. No names, shapes alone or pointer facts suffice.
                        eligible = (row_axis >= 0 and extents[row_axis] % rows == 0
                                    and row_axis in positions[0] and col_axis not in positions[0]
                                    and col_axis in positions[1] and row_axis not in positions[1])
                        owner = inner.operation
                        while eligible and owner is not None:
                            if any(a in owner.attributes for a in ("strictfp", "llvm.strictfp")):
                                eligible = False
                            owner = owner.parent
                    if eligible:
                        todo.append((inner, spec))
                    else:
                        walk(inner.operation)

    walk(module.operation)
    with ctx:
        for old, (positions, extents, mul_order, add_order) in todo:
            with old.location, _sc_ir.InsertionPoint(old):
                index = _sc_ir.IndexType.get()
                scalar = old.regions[0].blocks[0].arguments[0].type
                tensor = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _sc_ir.Operation.create(name, operands=list(operands), results=list(results),
                                                   attributes=attributes or {}, regions=regions)
                def constant(value):
                    return create("arith.constant", results=[index], attributes={
                        "value": _sc_ir.IntegerAttr.get(index, value)}).results[0]
                zero, one = constant(0), constant(1)
                limits = [constant(e) for e in extents]
                step = one if outputs == 1 else constant(outputs)
                offsets = [zero, *[constant(i) for i in range(1, outputs)]]
                row_offsets = [zero, *[constant(i) for i in range(1, rows)]]
                row_step = one if rows == 1 else constant(rows)

                def output_loop(depth, current, indices):
                    if depth < len(extents) - 1:
                        stride = (step if depth == len(extents) - 2 else
                                  row_step if rows != 1 and depth == len(extents) - 3 else one)
                        loop = create("scf.for", [zero, limits[depth], stride, current], [tensor], regions=1)
                        block = _sc_ir.Block.create_at_start(loop.regions[0], [index, tensor])
                        with _sc_ir.InsertionPoint(block):
                            updated = output_loop(depth + 1, block.arguments[1], [*indices, block.arguments[0]])
                            create("scf.yield", [updated])
                        return loop.results[0]
                    if rows == 1:
                        lanes = [indices]
                        for offset in offsets[1:]:
                            column = create("arith.addi", [indices[-1], offset], [index]).results[0]
                            lanes.append([*indices[:-1], column])
                    else:
                        lanes = []
                        for row_offset in row_offsets:
                            row = (indices[-2] if row_offset == zero else
                                   create("arith.addi", [indices[-2], row_offset], [index]).results[0])
                            for offset in offsets:
                                column = (indices[-1] if offset == zero else
                                          create("arith.addi", [indices[-1], offset], [index]).results[0])
                                lanes.append([*indices[:-2], row, column])
                    out_indices = [[lane[d] for d in positions[2]] for lane in lanes]
                    def reduction():
                        initial = [create("tensor.extract", [current, *idx], [scalar]).results[0]
                                   for idx in out_indices]
                        # SCF -> CF forwards LLVM-typed values; CF -> LLVM keeps the key.
                        # Use the final branch property's spelling: a retained namespaced
                        # llvm.loop_annotation is ignored by the installed translator.
                        attributes = ({"loop_annotation": _sc_ir.Attribute.parse(
                            "#llvm.loop_annotation<unroll = <count = " + str(reduction_unroll) + " : i32>>")}
                            if reduction_unroll else {})
                        lane_count = rows * outputs
                        loop = create("scf.for", [zero, limits[-1], one, *initial], [scalar] * lane_count,
                                      attributes=attributes, regions=1)
                        block = _sc_ir.Block.create_at_start(loop.regions[0], [index, *([scalar] * lane_count)])
                        with _sc_ir.InsertionPoint(block):
                            accumulated, shared = [], {}
                            for lane_number, lane in enumerate(lanes):
                                all_indices = [*lane, block.arguments[0]]
                                values = []
                                for i in range(2):
                                    if rows == 1:
                                        key = i if len(extents) - 2 not in positions[i] else None
                                    else:
                                        key = (i,
                                               lane_number // outputs if len(extents) - 3 in positions[i] else 0,
                                               lane_number % outputs if len(extents) - 2 in positions[i] else 0)
                                    if key is not None and key in shared:
                                        value = shared[key]
                                    else:
                                        value = create("tensor.extract", [old.operands[i], *[all_indices[d] for d in positions[i]]],
                                                       [scalar]).results[0]
                                        if key is not None:
                                            shared[key] = value
                                    values.append(value)
                                product = create("arith.mulf", [values[d] for d in mul_order], [scalar]).results[0]
                                operands = [product, block.arguments[lane_number + 1]]
                                accumulated.append(create("arith.addf", [operands[d] for d in add_order], [scalar]).results[0])
                            create("scf.yield", accumulated)
                        destination = current
                        for value, idx in zip(loop.results, out_indices):
                            destination = create("tensor.insert", [value, destination, *idx], [tensor]).results[0]
                        return destination
                    if output_guard is None:
                        return reduction()
                    condition = output_guard(old.operation, lanes, positions[2], create, constant)
                    guarded = create("scf.if", [condition], [tensor], regions=2)
                    for branch in range(2):
                        block = _sc_ir.Block.create_at_start(guarded.regions[branch], [])
                        with _sc_ir.InsertionPoint(block):
                            create("scf.yield", [reduction() if branch == 0 else current])
                    return guarded.results[0]

                result = output_loop(0, old.operands[2], [])
                replacement = result.owner
                for name in old.attributes:
                    if name.startswith("prov."):
                        replacement.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                text = _sc_ir.StringAttr(previous).value + "," if previous is not None else ""
                suffix = ("_2x4_outputs" if rows != 1 else
                          "" if outputs == 1 else "_" + str(outputs) + "_outputs")
                transforms = text + "scalar_contraction_accumulator" + suffix
                if reduction_unroll:
                    transforms += ",unroll_scalar_contraction_reduction_by_" + str(reduction_unroll)
                replacement.attributes["prov.transforms"] = _sc_ir.StringAttr.get(transforms)
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)


_SC_MARKER = "__merlin_scalar_contraction_accumulator__"
_SC2_MARKER = "__merlin_scalar_contraction_accumulator_2_outputs__"
_SC4_MARKER = "__merlin_scalar_contraction_accumulator_4_outputs__"
_SC8_MARKER = "__merlin_scalar_contraction_accumulator_8_outputs__"
_SC2X4_MARKER = "__merlin_scalar_contraction_accumulator_2x4_outputs__"
_SC_MARKERS = (_SC_MARKER, _SC2_MARKER, _SC4_MARKER, _SC8_MARKER, _SC2X4_MARKER)
_SC_UNROLL2_MARKER = "__merlin_scalar_contraction_reduction_unroll_2__"
_SC_UNROLL4_MARKER = "__merlin_scalar_contraction_reduction_unroll_4__"
_SC_UNROLL_MARKERS = {_SC_UNROLL2_MARKER: 2, _SC_UNROLL4_MARKER: 4}
_SC_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    hints = [m for m in _SC_UNROLL_MARKERS if m in passes]
    if len(hints) > 1:
        raise ValueError("scalar contraction reduction unroll counts are alternatives")
    unroll = _SC_UNROLL_MARKERS[hints[0]] if hints else 0
    passes = [p for p in passes if p not in _SC_UNROLL_MARKERS]
    selected = [m for m in _SC_MARKERS if m in passes]
    if not selected:
        return _SC_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if len(selected) != 1:
        raise ValueError("scalar contraction accumulator schedules are alternatives")
    i = passes.index(selected[0])
    head_generalizes = any('linalg-generalize-named-ops' in p for p in passes[:i])
    _SC_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), 0, (), (), (),
                        pre_generalize if head_generalizes else ())
    _SC_ORIG_RUN_STAGES(ctx, module, 'func.func(linalg-generalize-named-ops)', 0, (), (), (),
                        () if head_generalizes else pre_generalize)
    rows = 2 if selected[0] == _SC2X4_MARKER else 1
    outputs = {_SC_MARKER: 1, _SC2_MARKER: 2, _SC4_MARKER: 4, _SC8_MARKER: 8, _SC2X4_MARKER: 4}[selected[0]]
    name = 'scalar_contraction_accumulator' + ('_2x4_outputs' if rows != 1 else '' if outputs == 1 else '_' + str(outputs) + '_outputs')
    count = _scalarize_tensor_contractions(ctx, module, outputs, unroll, rows)
    print('OK', name, count)
    if unroll:
        print('OK unroll_scalar_contraction_reduction_by_' + str(unroll), count)
    _SC_ORIG_RUN_STAGES(ctx, module, ','.join(passes[i + 1:]), erase, mid, late, post_openmp, ())
"""


def apply_for_test(mlir_text: str, *, outputs: int = 1, reduction_unroll: int = 0, rows: int = 1) -> tuple[str, int]:
    """Run the actual shipped structural rewrite in the toolchain owning Python."""
    import subprocess
    import tempfile
    from pathlib import Path

    from .toolchain import m2m_python

    if type(outputs) is not int or outputs not in (1, 2, 4, 8):
        raise ValueError("scalar accumulator outputs must be 1, 2, 4 or 8")
    if type(reduction_unroll) is not int or reduction_unroll not in (0, 2, 4):
        raise ValueError("scalar accumulator reduction unroll must be 0, 2 or 4")
    if type(rows) is not int or rows not in (1, 2) or (rows == 2 and outputs != 4):
        raise ValueError("rectangular scalar accumulator schedule requires two rows and four columns")
    work = Path(tempfile.mkdtemp(prefix="merlin_scalar_contraction_"))
    src, script = work / "in.mlir", work / "run.py"
    src.write_text(mlir_text)
    script.write_text(
        "import sys\nfrom torch_mlir import ir\n"
        + RUNNER_PRELUDE.split("_SC_MARKER =", 1)[0]
        + "ctx = ir.Context()\nmodule = ir.Module.parse(open(sys.argv[1]).read(), ctx)\n"
        + f"print('COUNT', _scalarize_tensor_contractions(ctx, module, {outputs}, {reduction_unroll}, {rows}))\nprint('MODULE_BEGIN')\nprint(module)\n"
    )
    proc = subprocess.run([str(m2m_python()), str(script), str(src)], capture_output=True, text=True, timeout=120)
    if proc.returncode:
        raise RuntimeError(f"scalar contraction rewrite failed:\n{proc.stdout}\n{proc.stderr}")
    count = int(next(x.split()[1] for x in proc.stdout.splitlines() if x.startswith("COUNT ")))
    return proc.stdout.split("MODULE_BEGIN\n", 1)[1], count
