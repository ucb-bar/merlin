"""Weights the model re-lays-out at run time, and whether that work can be hoisted offline.

A `linalg.transpose` whose input is a function argument is not computation — it is the model paying,
on every inference, to convert a weight into the layout its consumer wanted all along. The packer
could have stored it that way once.

MEASURED, and why this module exists. On `gemma2_2b_int8_full_seq8` 183 such transposes move
**2,493.0 MiB per inference** against a total weight blob of 2,505 MiB: essentially every weight in
the model, re-laid-out every time it runs. It is not only a throughput tax. The largest one, the
562.5 MiB int8 tied head, is what killed a whole-model FireSim run — `FAIL alloc bytes=589824064` at
op 11,494 of 11,526, with the arena unable to serve it. Hoisting all 183 into the bundle took the
count to 0 with the MAC total and the matrix-unit routing both unchanged.

HOISTABLE MEANS SOLE-USE. Storing a weight pre-transposed is safe only when the transpose is that
argument's ONLY consumer. If anything else reads the argument, it would silently start seeing
transposed data. :func:`weight_layout_report` therefore splits the two cases and never merges them —
on the model above the split was 183 sole-use and 0 mixed, but a model with a weight used both
transposed and untransposed must not be hoisted blindly.

The rewrite itself is bit-exact: a transpose moves elements, it does not compute. Every value the
consumer sees is the value it saw before, at the same index, so goldens survive untouched.

This is a structural opportunity analysis, not permission to change an argument's stored bytes.
The caller must separately establish immutable invocation ownership and storage identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any

from xdsl.dialects import builtin, func, tensor
from xdsl.dialects.linalg.ops import TransposeOp, YieldOp

#: bytes per element, by the dtype spelling `mlir_query.type_shape_dtype` returns.
_WIDTH = {"i8": 1, "u8": 1, "ui8": 1, "i16": 2, "f16": 2, "bf16": 2, "i32": 4, "f32": 4, "i64": 8, "f64": 8}


@dataclass(frozen=True)
class WeightLayoutStep:
    """One checked, row-major, element-preserving source layout operation."""

    operation: str
    shape: tuple[int, ...]
    result_shape: tuple[int, ...]
    permutation: tuple[int, ...] = ()
    reassociation: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class WeightRelayout:
    """A transpose and its maximal sole-use static layout chain from an argument."""

    arg: int
    shape: list[int]
    dtype: str
    result_shape: list[int]
    permutation: tuple[int, ...]
    steps: tuple[WeightLayoutStep, ...]
    #: True when the transpose is the argument's only consumer, so the layout can be pre-applied.
    hoistable: bool
    #: why not, when it is not
    reason: str = ""

    @property
    def bytes_moved(self) -> int:
        width = _WIDTH.get(self.dtype)
        if width is None:
            return 0
        return sum(prod(step.result_shape) * width for step in self.steps if step.operation == "linalg.transpose")

    @property
    def transposes(self) -> int:
        return sum(step.operation == "linalg.transpose" for step in self.steps)


@dataclass
class WeightLayoutReport:
    relayouts: list[WeightRelayout] = field(default_factory=list)
    #: transposes whose element width is unknown, so their cost could not be priced. Surfaced rather
    #: than counted as zero: an unpriceable op must not read as a free one.
    unpriceable: list[str] = field(default_factory=list)

    @property
    def hoistable(self) -> list[WeightRelayout]:
        return [r for r in self.relayouts if r.hoistable]

    @property
    def blocked(self) -> list[WeightRelayout]:
        return [r for r in self.relayouts if not r.hoistable]

    @property
    def hoistable_bytes(self) -> int:
        """Weight traffic per inference that storing the transposed layout would remove."""
        return sum(r.bytes_moved for r in self.hoistable)

    @property
    def total_bytes(self) -> int:
        return sum(r.bytes_moved for r in self.relayouts)


def _func(module: Any, func_name: str):
    """Select one exact top-level symbol, never a nested function or substring."""
    if not isinstance(module, builtin.ModuleOp) or type(func_name) is not str or not func_name:
        raise ValueError("weight layout requires a module and an exact top-level function name")
    matches = [op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == func_name]
    if len(matches) != 1 or len(matches[0].body.blocks) != 1:
        raise ValueError(f"weight layout requires one defined top-level function {func_name!r}")
    return matches[0]


def _tensor(type):
    return (
        isinstance(type, builtin.TensorType)
        and isinstance(type.encoding, builtin.NoneAttr)
        and all(d >= 0 for d in type.get_shape())
    )


def _transpose_problem(op, arg, fn):
    if op.parent is not fn.body.block:
        return "transpose is not directly owned by the selected function"
    if len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1 or op.inputs[0] is not arg:
        return "argument must be operand zero of one tensor transpose"
    source, result = arg.type, op.results[0].type
    if not _tensor(source) or not _tensor(result) or op.outputs[0].type != result:
        return "transpose requires static unencoded ranked tensor input, init and result"
    permutation = tuple(op.permutation.get_values())
    rank = len(source.get_shape())
    if rank == 0 or sorted(permutation) != list(range(rank)):
        return "transpose requires an explicit complete rank-N permutation"
    if (
        source.element_type != result.element_type
        or tuple(source.get_shape()[i] for i in permutation) != result.get_shape()
    ):
        return "transpose permutation, shape or element types disagree"
    body = op.body.block
    operations = list(body.ops)
    if (
        len(body.args) != 2
        or any(a.type != source.element_type for a in body.args)
        or len(operations) != 1
        or not isinstance(operations[0], YieldOp)
        or tuple(operations[0].operands) != (body.args[0],)
        or set(op.properties) != {"permutation"}
        or any(not key.startswith("prov.") for operation in (op, operations[0]) for key in operation.attributes)
        or operations[0].properties
    ):
        return "transpose has unsupported executable fields or scalar body"
    return ""


def _step(op, value, fn):
    if isinstance(op, TransposeOp):
        if _transpose_problem(op, value, fn):
            return None
        return WeightLayoutStep(
            op.name, value.type.get_shape(), op.results[0].type.get_shape(), tuple(op.permutation.get_values())
        )
    if not isinstance(op, tensor.CollapseShapeOp | tensor.ExpandShapeOp):
        return None
    if (
        op.parent is not fn.body.block
        or len(op.operands) != 1
        or op.operands[0] is not value
        or not _tensor(value.type)
        or not _tensor(op.results[0].type)
        or value.type.element_type != op.results[0].type.element_type
        or any(not key.startswith("prov.") for key in op.attributes)
    ):
        return None
    source, result = value.type.get_shape(), op.results[0].type.get_shape()
    groups = tuple(tuple(axis.value.data for axis in group) for group in op.reassociation)
    if isinstance(op, tensor.CollapseShapeOp):
        allowed, expanded, collapsed = {"reassociation"}, source, result
    else:
        allowed, expanded, collapsed = {"reassociation", "static_output_shape"}, result, source
        if tuple(op.static_output_shape.get_values()) != result:
            return None
    if (
        set(op.properties) != allowed
        or not source
        or not result
        or len(groups) != len(collapsed)
        or any(not group for group in groups)
        or tuple(axis for group in groups for axis in group) != tuple(range(len(expanded)))
        or tuple(prod(expanded[axis] for axis in group) for group in groups) != collapsed
    ):
        return None
    return WeightLayoutStep(op.name, source, result, reassociation=groups)


def layout_chain(value, fn):
    """The maximal supported sole-use chain; unknown operations and extra readers stop it."""
    operations, steps = [], []
    while len(uses := list(value.uses)) == 1 and uses[0].index == 0:
        op = uses[0].operation
        step = _step(op, value, fn)
        if step is None:
            break
        operations.append(op)
        steps.append(step)
        value = op.results[0]
    return tuple(operations), tuple(steps)


def apply_layout_steps(value, steps):
    """Evaluate checked views/permutations, as group_prepack does, without numeric conversion.

    The caller chooses the element representation. Capture rewriting uses fixed-width raw
    byte records, so BF16 and floating NaN payloads never pass through arithmetic.
    """
    import numpy as np

    for step in steps:
        if tuple(value.shape) != step.shape:
            raise ValueError("stored tensor shape differs from the checked layout step")
        if step.operation == "linalg.transpose":
            if sorted(step.permutation) != list(range(value.ndim)):
                raise ValueError("invalid stored tensor permutation")
            value = np.transpose(value, step.permutation)
        elif step.operation in {"tensor.collapse_shape", "tensor.expand_shape"}:
            value = value.reshape(step.result_shape)
        else:
            raise ValueError("unsupported stored tensor layout step")
        if tuple(value.shape) != step.result_shape:
            raise ValueError("stored tensor layout did not produce the checked shape")
    return np.ascontiguousarray(value)


def weight_layout_report(module: Any, func_name: str = "forward") -> WeightLayoutReport:
    """Every run-time re-layout of a weight in `func_name`, split by whether it can be hoisted."""
    try:
        module.verify()
    except Exception as exc:
        raise ValueError(f"weight layout requires verified source IR: {exc}") from exc
    fn = _func(module, func_name)
    rep = WeightLayoutReport()

    for i, arg in enumerate(fn.body.blocks[0].args):
        uses = list(arg.uses)
        transposes = [u for u in uses if isinstance(u.operation, TransposeOp)]
        if not transposes:
            continue
        op = transposes[0].operation
        shape = list(arg.type.get_shape()) if isinstance(arg.type, builtin.TensorType) else []
        dtype = str(arg.type.element_type) if isinstance(arg.type, builtin.TensorType) else "unknown"
        result = op.results[0].type if op.results else None
        res_shape = list(result.get_shape()) if isinstance(result, builtin.TensorType) else []

        if dtype not in _WIDTH:
            rep.unpriceable.append(f"arg {i}: unknown element width for dtype {dtype!r}")

        sole = len(uses) == 1 and len(transposes) == 1 and transposes[0].index == 0
        reason = (
            _transpose_problem(op, arg, fn)
            if sole
            else (
                f"argument has {len(uses)} consumers ({len(transposes)} transpose(s)); pre-transposing "
                "would change what the other readers see"
            )
        )
        first = _step(op, arg, fn)
        _operations, steps = layout_chain(arg, fn) if sole and not reason else ((), (first,) if first else ())
        if steps:
            res_shape = list(steps[-1].result_shape)
        rep.relayouts.append(
            WeightRelayout(
                arg=i,
                shape=shape,
                dtype=dtype,
                result_shape=res_shape,
                permutation=tuple(op.permutation.get_values()),
                steps=steps,
                hoistable=sole and not reason,
                reason=reason,
            )
        )
    return rep
