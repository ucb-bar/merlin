"""Explicit closed-mask observation scheduling for ordered tensor contractions.

Only tiles whose every contraction output is discarded by a proved source
select omit their reduction. Active and partial tiles retain every original
multiply/add and increasing reduction order. This grants no numeric permission:
the caller must explicitly allow nontrapping arithmetic and unobserved flags.
Tensor SSA and upstream bufferization retain ownership and alias authority.
"""

from __future__ import annotations

from dataclasses import dataclass

FEATURE = "scalar_contraction_masked_observation"
MARKER = "__merlin_scalar_contraction_masked_observation__"


def _edit_pipeline(passes: list[str]) -> list[str]:
    from .scalar_contraction import (
        EIGHT_OUTPUTS_MARKER as eight,
    )
    from .scalar_contraction import (
        FOUR_OUTPUTS_MARKER as four,
    )
    from .scalar_contraction import (
        MARKER as one,
    )
    from .scalar_contraction import (
        RECTANGULAR_MARKER as rectangle,
    )
    from .scalar_contraction import (
        TWO_OUTPUTS_MARKER as two,
    )

    selected = [i for i, p in enumerate(passes) if p in (one, two, four, eight, rectangle)]
    if len(selected) != 1 or MARKER in passes:
        raise ValueError("closed-mask scheduling requires one explicit scalar contraction schedule")
    i = selected[0]
    return [*passes[:i], MARKER, *passes[i:]]


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register
    from .scalar_contraction import _OUTPUTS_FEATURES

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description="Skip reductions only for fully source-discarded output tiles under an explicit nontrapping, unobserved-floating-flags contract.",
                edit_pipeline=_edit_pipeline,
                requires_exactly_one_of=_OUTPUTS_FEATURES,
            )
        )
    return FEATURE


@dataclass(frozen=True)
class MaskEffectContract:
    nontrapping: bool
    floating_flags_unobserved: bool

    def validate(self) -> None:
        if self.nontrapping is not True or self.floating_flags_unobserved is not True:
            raise ValueError("masked contraction requires explicit nontrapping/unobserved flags")


RUNNER_PRELUDE = r"""
def _mc_shape(value, element="f32"):
    from torch_mlir import ir
    if not isinstance(value.type, ir.RankedTensorType):
        return None
    tensor = ir.RankedTensorType(value.type)
    shape = list(tensor.shape)
    if str(tensor.element_type) != element or any(d < 0 for d in shape):
        return None
    return shape

def _mc_attrs(op, allowed):
    return all(name in allowed or name.startswith("prov.") for name in op.attributes)

def _mc_single(value):
    uses = list(value.uses)
    return uses[0].owner.operation if len(uses) == 1 else None

def _mc_maps(op):
    from torch_mlir import ir
    if "indexing_maps" not in op.attributes:
        return None
    maps = [ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if not maps or any(m.n_symbols or m.n_dims != maps[0].n_dims for m in maps):
        return None
    result = []
    for affine in maps:
        row = []
        for expr in affine.results:
            if isinstance(expr, ir.AffineDimExpr):
                row.append(("dim", ir.AffineDimExpr(expr).position))
            elif isinstance(expr, ir.AffineConstantExpr) and ir.AffineConstantExpr(expr).value == 0:
                row.append(("zero", 0))
            else:
                return None
        result.append(row)
    return maps[0].n_dims, result

def _mc_parallel(op):
    from torch_mlir import ir
    if (op.name != "linalg.generic" or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1 or not _mc_attrs(op,
                {"indexing_maps", "iterator_types", "operandSegmentSizes"})):
        return None
    maps = _mc_maps(op)
    if maps is None:
        return None
    dims, positions = maps
    if [str(i) for i in op.attributes["iterator_types"]] != ["#linalg.iterator_type<parallel>"] * dims:
        return None
    if positions[-1] != [("dim", d) for d in range(dims)]:
        return None
    if list(op.regions[0].blocks[0].arguments[-1].uses):
        return None
    return positions

def _mc_before(value, operation):
    from torch_mlir import ir
    if isinstance(value, ir.BlockArgument):
        # The exact enclosing block's argument dominates; nested capture is
        # deliberately unsupported rather than inventing dominance.
        return value.owner == operation.block
    owner = value.owner.operation
    if owner.block != operation.block:
        return False
    for current in operation.block.operations:
        if current.operation == owner:
            return True
        if current.operation == operation:
            return False
    return False

def _mc_mask_source(value, operation):
    from torch_mlir import ir
    if _mc_before(value, operation):
        return value, False
    if isinstance(value, ir.BlockArgument):
        return None
    invert = value.owner.operation
    if not isinstance(invert, ir.Operation) or invert.block != operation.block:
        return None
    positions = _mc_parallel(invert)
    if (positions is None or len(invert.operands) != 2 or positions[0] != positions[1]
            or _mc_shape(value, "i1") != _mc_shape(invert.operands[0], "i1")):
        return None
    block = invert.regions[0].blocks[0]
    body = list(block.operations)
    if ([o.operation.name for o in body] != ["arith.constant", "arith.xori", "linalg.yield"]
            or any(not _mc_attrs(o.operation, {"value"}) for o in body[:1])
            or any(not _mc_attrs(o.operation, set()) for o in body[1:])):
        return None
    literal, xor, yield_op = body
    if (str(literal.results[0].type) != "i1" or ir.IntegerAttr(literal.attributes["value"]).value == 0
            or list(xor.operands) not in ([block.arguments[0], literal.results[0]],
                                        [literal.results[0], block.arguments[0]])
            or list(yield_op.operands) != [xor.results[0]]
            or not _mc_before(invert.operands[0], operation)):
        return None
    return invert.operands[0], True

def _mc_spec(op):
    from torch_mlir import ir
    source = _scalar_contraction_spec(op)
    if source is None or not _mc_attrs(op, {"indexing_maps", "iterator_types", "operandSegmentSizes"}):
        return None
    owner = op
    while owner is not None:
        if any("strictfp" in name or "strictfp" in str(owner.attributes[name]) for name in owner.attributes):
            return None
        owner = owner.parent
    value = op.results[0]
    original_shape = _mc_shape(value)
    if original_shape is None or any(d == 0 for d in original_shape):
        return None
    current_shape = original_shape
    consumer = _mc_single(value)
    # Support exact full row-major flatten/unflatten. General reassociation or
    # permutation refuses; total element equality alone is insufficient.
    while consumer is not None and consumer.name in ("tensor.collapse_shape", "tensor.expand_shape"):
        if (len(consumer.operands) != 1 or len(consumer.results) != 1
                or consumer.block != op.block
                or not _mc_attrs(consumer, {"reassociation", "static_output_shape"})):
            return None
        next_shape = _mc_shape(consumer.results[0])
        if next_shape is None or any(d == 0 for d in next_shape):
            return None
        reassociation = [[ir.IntegerAttr(i).value for i in group]
                         for group in consumer.attributes["reassociation"]]
        larger = len(current_shape) if consumer.name == "tensor.collapse_shape" else len(next_shape)
        smaller = len(next_shape) if consumer.name == "tensor.collapse_shape" else len(current_shape)
        if smaller != 1 or reassociation != [list(range(larger))]:
            return None
        product = lambda s: __import__("math").prod(s)
        if product(current_shape) != product(next_shape):
            return None
        value, current_shape = consumer.results[0], next_shape
        consumer = _mc_single(value)
    positions = _mc_parallel(consumer) if consumer is not None else None
    if positions is None or len(consumer.operands) != 3 or _mc_shape(consumer.results[0]) != current_shape:
        return None
    arguments = list(consumer.regions[0].blocks[0].arguments)
    body = list(consumer.regions[0].blocks[0].operations)
    if ([o.operation.name for o in body] != ["arith.mulf", "linalg.yield"]
            or any(not _mc_attrs(o.operation, {"fastmath"}) for o in body)
            or any("fastmath" in o.attributes and str(o.attributes["fastmath"]) != "#arith.fastmath<none>" for o in body)
            or list(body[1].operands) != [body[0].results[0]]):
        return None
    lane = next((i for i in range(2) if consumer.operands[i] == value), None)
    if (lane is None or positions[lane] != positions[-1]
            or list(body[0].operands) not in ([arguments[0], arguments[1]], [arguments[1], arguments[0]])):
        return None
    # Source coefficient is a rank-zero tensor. It remains computed at its
    # original position; no division/scale rewrite or assumption is granted.
    if _mc_shape(consumer.operands[1-lane]) != [] or positions[1-lane] != []:
        return None
    scaled = consumer.results[0]
    select = _mc_single(scaled)
    positions = _mc_parallel(select) if select is not None else None
    if (positions is None or select.block != op.block or len(select.operands) != 4
            or _mc_shape(select.results[0]) != current_shape):
        return None
    block = select.regions[0].blocks[0]
    body = list(block.operations)
    if ([o.operation.name for o in body] != ["arith.select", "linalg.yield"]
            or any(not _mc_attrs(o.operation, set()) for o in body)
            or list(body[1].operands) != [body[0].results[0]]):
        return None
    choices = list(body[0].operands)
    if any(v not in list(block.arguments)[:3] for v in choices) or choices[1] == choices[2]:
        return None
    condition = list(block.arguments).index(choices[0])
    chosen = next((i for i in (1,2) if select.operands[list(block.arguments).index(choices[i])] == scaled), None)
    if condition is None or chosen is None:
        return None
    scaled_position = list(block.arguments).index(choices[chosen])
    if positions[scaled_position] != positions[-1]:
        return None
    mask = select.operands[condition]
    mask_shape = _mc_shape(mask, "i1")
    if mask_shape is None or len(mask_shape) != len(positions[condition]):
        return None
    for mapping, extent in zip(positions[condition], mask_shape):
        if (mapping[0] == "zero" and extent != 1
                or mapping[0] == "dim" and extent != current_shape[mapping[1]]):
            return None
    resolved = _mc_mask_source(mask, op)
    if resolved is None:
        return None
    raw, inverted = resolved
    return raw, positions[condition], current_shape, (chosen == 1) != inverted

def _scalarize_closed_mask_contractions(ctx, module, *, outputs=1, rows=1, reduction_unroll=0):
    from torch_mlir import ir
    plans = []
    for op in module.body.operations:
        def walk(operation):
            for region in operation.regions:
                for block in region.blocks:
                    for child in list(block.operations):
                        spec = _mc_spec(child.operation)
                        if spec is not None:
                            plans.append((child.operation, spec))
                        else:
                            walk(child.operation)
        walk(op.operation)
    def guard(operation, lanes, output_positions, create, constant):
        raw, mask_positions, observed_shape, true_observed = next(spec for old, spec in plans if old == operation)
        source_shape = _mc_shape(operation.results[0])
        conditions = []
        for lane in lanes:
            source_indices = [lane[d] for d in output_positions]
            flat = source_indices[0]
            for coordinate, extent in zip(source_indices[1:], source_shape[1:]):
                flat = create("arith.addi", [create("arith.muli", [flat, constant(extent)],
                    [ir.IndexType.get()]).results[0], coordinate], [ir.IndexType.get()]).results[0]
            observed_indices = []
            for number, extent in enumerate(observed_shape):
                stride = __import__("math").prod(observed_shape[number+1:])
                coordinate = flat if stride == 1 else create("arith.divui", [flat, constant(stride)], [ir.IndexType.get()]).results[0]
                if number != 0:
                    coordinate = create("arith.remui", [coordinate, constant(extent)], [ir.IndexType.get()]).results[0]
                observed_indices.append(coordinate)
            coordinates = [constant(0) if kind == "zero" else observed_indices[position] for kind, position in mask_positions]
            condition = create("tensor.extract", [raw, *coordinates], [ir.IntegerType.get_signless(1)]).results[0]
            if not true_observed:
                one = create("arith.constant", results=[ir.IntegerType.get_signless(1)],
                    attributes={"value": ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 1)}).results[0]
                condition = create("arith.xori", [condition, one], [ir.IntegerType.get_signless(1)]).results[0]
            conditions.append(condition)
        combined = conditions[0]
        for condition in conditions[1:]:
            combined = create("arith.ori", [combined, condition], [ir.IntegerType.get_signless(1)]).results[0]
        return combined
    return _scalarize_tensor_contractions(ctx, module, outputs=outputs, rows=rows,
        reduction_unroll=reduction_unroll, selection=[op for op,_ in plans], output_guard=guard)
"""

STAGE_RUNNER = r"""
_MC_MARKER = "__merlin_scalar_contraction_masked_observation__"
_MC_ORIG_RUN_STAGES = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _MC_MARKER not in passes:
        return _MC_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if len(sys.argv) <= 20 or sys.argv[20] != '1':
        raise ValueError("closed-mask scheduling requires explicit arithmetic effect permission")
    selected = [m for m in _SC_MARKERS if m in passes]
    if len(selected) != 1 or passes.count(_MC_MARKER) != 1:
        raise ValueError("closed-mask scheduling requires one scalar schedule")
    outputs = {_SC_MARKER: 1, _SC2_MARKER: 2, _SC4_MARKER: 4, _SC8_MARKER: 8, _SC2X4_MARKER: 4}[selected[0]]
    rows = 2 if selected[0] == _SC2X4_MARKER else 1
    hints = [m for m in _SC_UNROLL_MARKERS if m in passes]
    if len(hints) > 1:
        raise ValueError("scalar contraction reduction unroll counts are alternatives")
    count = _scalarize_closed_mask_contractions(ctx, module, outputs=outputs, rows=rows,
        reduction_unroll=_SC_UNROLL_MARKERS[hints[0]] if hints else 0)
    print('OK scalar_contraction_masked_observation', count)
    return _MC_ORIG_RUN_STAGES(ctx, module, ','.join(p for p in passes if p != _MC_MARKER),
        erase, mid, late, post_openmp, pre_generalize)
"""


def apply_for_test(mlir_text: str, *, effects: MaskEffectContract, outputs: int = 1, rows: int = 1) -> tuple[str, int]:
    """Run the source-derived prototype in the authoritative upstream IR tool."""
    import subprocess
    import tempfile
    from pathlib import Path

    from .scalar_contraction import RUNNER_PRELUDE as scalar_prelude
    from .toolchain import m2m_python

    if not isinstance(effects, MaskEffectContract):
        raise ValueError("explicit MaskEffectContract is required")
    effects.validate()
    if (
        type(outputs) is not int
        or type(rows) is not int
        or outputs not in (1, 2, 4, 8)
        or rows not in (1, 2)
        or (rows == 2 and outputs != 4)
    ):
        raise ValueError("invalid scalar output tile")
    with tempfile.TemporaryDirectory(prefix="merlin_masked_contraction_") as directory:
        root = Path(directory)
        source, script = root / "source.mlir", root / "run.py"
        source.write_text(mlir_text)
        script.write_text(
            "import sys\nfrom torch_mlir import ir\n"
            + scalar_prelude.split("_SC_MARKER =", 1)[0]
            + RUNNER_PRELUDE
            + "\nctx=ir.Context()\nmodule=ir.Module.parse(open(sys.argv[1]).read(),ctx)\n"
            + f"print('COUNT',_scalarize_closed_mask_contractions(ctx,module,outputs={outputs},rows={rows}))\n"
            + "print('MODULE_BEGIN')\nprint(module)\n"
        )
        result = subprocess.run(
            [str(m2m_python()), str(script), str(source)], capture_output=True, text=True, timeout=120, check=False
        )
        if result.returncode:
            raise RuntimeError(f"masked contraction rewrite failed:\n{result.stdout}\n{result.stderr}")
        count = int(next(line.split()[1] for line in result.stdout.splitlines() if line.startswith("COUNT ")))
        return result.stdout.split("MODULE_BEGIN\n", 1)[1], count


def require_report(stdout: str, workdir) -> int:
    """Require evidence that the ordinary runner executed this explicit policy."""
    import json
    from pathlib import Path

    tokens = [line.split() for line in stdout.splitlines() if line.startswith("OK " + FEATURE + " ")]
    if len(tokens) != 1 or len(tokens[0]) != 3:
        raise ValueError("closed-mask runner report is absent or ambiguous")
    try:
        count = int(tokens[0][2])
    except ValueError as exc:
        raise ValueError("invalid closed-mask runner count") from exc
    if count < 0:
        raise ValueError("invalid closed-mask runner count")
    record = {
        "schema": "merlin.closed_mask_contraction.v1",
        "policy": FEATURE,
        "rewritten_contractions": count,
        "effect_permissions": {"nontrapping": True, "floating_flags_unobserved": True},
        "scope": "Immediate typed all-use observation closure; partial/active tiles retain source arithmetic. Runtime omitted-work counts and performance are not inferred.",
    }
    (Path(workdir) / "masked_contraction_report.json").write_text(json.dumps(record, indent=2) + "\n")
    return count
