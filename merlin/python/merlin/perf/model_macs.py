"""Full-source MAC domains, not the narrower matmul-specialization shape space.

A MAC requires an actual yielded multiply-accumulate recurrence. Static affine
domains can include several reduction axes, as in convolution. Unsupported
recurrences/domains/control flow remain visible UNKNOWN work, never missing zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from types import SimpleNamespace
from typing import Any

from xdsl.ir import Operation
from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr

from merlin.common.mlir_query import parse
from merlin.perf.compiler_plan_evidence import _source_has_multiply_accumulate


@dataclass(frozen=True)
class ModelMACShape:
    op: str
    parallel: tuple[int, ...] = ()
    reduction: tuple[int, ...] = ()
    dtypes: tuple[str, ...] = ()
    status: str = "UNKNOWN"
    reason: str = "unproved source MAC domain"
    source_op_index: int | None = None

    @property
    def macs(self) -> int | None:
        return prod(self.parallel+self.reduction) if self.status == "derived" else None


def _iterators(op):
    try:
        return tuple(item.data.value for item in op.get_iterator_types())
    except (AttributeError, ValueError, NotImplementedError):
        return None


def _yield_uses_product(op):
    """Unknown product-bearing reductions are not assumed to be non-MAC work."""
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return False
    terminator = op.regions[0].block.last_op
    if terminator is None:
        return False
    pending, seen = list(terminator.operands), set()
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        owner = value.owner
        if isinstance(owner, Operation):
            if owner.name in {"arith.muli", "arith.mulf", "math.fma"}:
                return True
            pending.extend(owner.operands)
    return False


def _interval(expr, bounds):
    if isinstance(expr, AffineConstantExpr):
        return expr.value, expr.value
    if isinstance(expr, AffineDimExpr):
        return 0, bounds[expr.position]-1
    if not isinstance(expr, AffineBinaryOpExpr):
        raise ValueError("symbolic/unsupported affine address")
    left, right = _interval(expr.lhs, bounds), _interval(expr.rhs, bounds)
    if expr.kind == AffineBinaryOpKind.Add:
        return left[0]+right[0], left[1]+right[1]
    if expr.kind == AffineBinaryOpKind.Mul:
        if left[0] != left[1] and right[0] != right[1]:
            raise ValueError("nonlinear affine address")
        corners = [a*b for a in left for b in right]
        return min(corners), max(corners)
    if right[0] != right[1] or right[0] <= 0:
        raise ValueError("unsupported affine divisor")
    divisor = right[0]
    if expr.kind == AffineBinaryOpKind.FloorDiv:
        return left[0]//divisor, left[1]//divisor
    if expr.kind == AffineBinaryOpKind.CeilDiv:
        return -(-left[0]//divisor), -(-left[1]//divisor)
    if expr.kind == AffineBinaryOpKind.Mod:
        return 0, divisor-1
    raise ValueError("unsupported affine expression")


def _domain(op, iterators):
    if any(kind not in {"parallel", "reduction"} for kind in iterators):
        raise ValueError("unsupported iterator kind")
    maps = tuple(attr.data for attr in op.get_indexing_maps())
    if len(maps) != len(op.operands) or not maps:
        raise ValueError("indexing maps do not cover every operand")
    if any(mapping.num_dims != len(iterators) or mapping.num_symbols for mapping in maps):
        raise ValueError("symbolic or inconsistent affine domain")
    shapes, dtypes = [], []
    bounds = [None]*len(iterators)
    for value, mapping in zip(op.operands, maps, strict=True):
        shape = tuple(value.type.get_shape())
        if len(shape) != len(mapping.results) or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError("dynamic/empty or mismatched operand shape")
        shapes.append(shape)
        dtypes.append(str(value.type.get_element_type()))
        for extent, expr in zip(shape, mapping.results, strict=True):
            if isinstance(expr, AffineDimExpr):
                previous = bounds[expr.position]
                if previous is not None and previous != extent:
                    raise ValueError("conflicting projected iteration bounds")
                bounds[expr.position] = extent
    if any(bound is None for bound in bounds):
        raise ValueError("iteration bound lacks an exact operand dimension projection")
    # The output must index parallel coordinates only. This separates a true
    # accumulation from a reduction-labelled pointwise update of distinct cells.
    output_maps = maps[len(tuple(op.inputs)):]
    parallel = {i for i, kind in enumerate(iterators) if kind == "parallel"}
    if len(output_maps) != 1 or any(not isinstance(expr, AffineDimExpr) for expr in output_maps[0].results):
        raise ValueError("unsupported accumulation output map")
    projected = [expr.position for expr in output_maps[0].results]
    if len(projected) != len(set(projected)) or set(projected) != parallel:
        raise ValueError("output is not a permutation of the parallel domain")
    for mapping, shape in zip(maps, shapes, strict=True):
        for expr, extent in zip(mapping.results, shape, strict=True):
            lower, upper = _interval(expr, bounds)
            if lower < 0 or upper >= extent:
                raise ValueError("affine access is not proved in bounds on the inferred domain")
    return (tuple(bounds[i] for i, kind in enumerate(iterators) if kind == "parallel"),
            tuple(bounds[i] for i, kind in enumerate(iterators) if kind == "reduction"), tuple(dtypes))


def observe_model_macs(src: Any, *, entry: str | None = None) -> list[tuple[Any, ModelMACShape]]:
    """Observe one complete entry; return explicit UNKNOWN rows for missing coverage.

    Source operations are counted in original entry order. Defined helper calls
    and enclosing control flow need a separate invocation-multiplicity proof;
    this observer does not count every helper definition once and call it E2E.
    """
    try:
        module = parse(src)
        functions = [op for op in module.body.block.ops if op.name == "func.func" and op.body.blocks]
        matches = ([op for op in functions if op.sym_name.data == entry] if entry is not None else functions)
        if len(matches) != 1 or len(matches[0].body.blocks) != 1:
            raise ValueError("one defined single-block entry is required")
        entry_function = matches[0]
    except Exception as error:
        return [(None, ModelMACShape("unparsed_source", reason=str(error)))]
    top = {op: i for i, op in enumerate(entry_function.body.block.ops) if op.name != "func.return"}
    found = []
    for op in entry_function.walk():
        if op.name in {"func.call", "func.call_indirect"}:
            found.append((op, ModelMACShape(op.name, reason="called source work/multiplicity is unproved",
                                           source_op_index=top.get(op))))
            continue
        if not op.name.startswith("linalg."):
            continue
        iterators = _iterators(op)
        if not iterators:
            if _yield_uses_product(op):
                found.append((op, ModelMACShape(op.name, reason="product-bearing linalg work has unreadable iterators",
                                               source_op_index=top.get(op))))
            continue
        if "reduction" not in iterators:
            continue
        recurrence = _source_has_multiply_accumulate(op)
        if not recurrence and op.name != "linalg.generic":
            # Named linalg operations expose the same scalar region/iterator
            # interface. Reuse exactly the shared recurrence matcher, without
            # modifying source or widening the kernel specialization contract.
            recurrence = _source_has_multiply_accumulate(SimpleNamespace(name="linalg.generic",
                regions=op.regions, attributes={}, properties={"iterator_types": op.get_iterator_types()}))
        if not recurrence and not _yield_uses_product(op):
            continue
        try:
            if not recurrence:
                raise ValueError("product-bearing reduction has an unsupported yielded recurrence")
            if op not in top:
                raise ValueError("enclosing source control-flow multiplicity is unproved")
            parallel, reduction, dtypes = _domain(op, iterators)
            shape = ModelMACShape(op.name, parallel, reduction, dtypes, "derived",
                                  "one proved yielded MAC per static affine-domain point", top[op])
        except (ValueError, AttributeError, IndexError, NotImplementedError) as error:
            shape = ModelMACShape(op.name, reason=str(error), source_op_index=top.get(op))
        found.append((op, shape))
    return found
