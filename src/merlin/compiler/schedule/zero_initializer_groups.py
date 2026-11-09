"""Find closed source initialization groups without selecting a target schedule.

An exact zero splat used only as a signed contraction's seed can remain virtual
when the target independently realizes zero accumulation and complete readout.
These source proposals grant no hardware, ownership or numerical authority.
The selected shared-library releases and their import rosters stay unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from io import StringIO

from xdsl.dialects import arith, builtin, func
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.printer import Printer


def _digest(module):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return hashlib.sha256(stream.getvalue().encode()).hexdigest()


def _name(operation):
    return getattr(operation.attributes.get("op_name__"), "data", operation.name)


def _provenance_only(operation, *, properties=()):
    return (
        all(name.startswith("prov.") or name == "op_name__" for name in operation.attributes)
        and not set(operation.properties) - set(properties)
    )


@dataclass(frozen=True)
class ZeroInitializerGroup:
    """Current source operations that one checked device task may own together."""

    source_ir_sha256: str
    source_op_indices: tuple[int, ...]
    contraction_ordinal: int
    initializer_ordinal: int
    virtual_result_positions: tuple[tuple[int, int], ...]
    geometry_m_k_n: tuple[int, int, int]
    initializer_elements: int
    initializer_bytes: int

    def record(self):
        return {
            "schema": "merlin.zero_initializer_group.v1",
            **asdict(self),
            "required_target_realization": [
                "exact zero accumulator initialization",
                "complete original signed products and accumulator continuation",
                "complete source output readout and completion",
                "no physical read of the virtual initializer",
                "original external buffer ownership and alias checks",
            ],
            "cost_scope": "logical host materialization avoided; not physical traffic or cycles",
            "authority": "source grouping proposal only; original final execution checks required",
        }

    def verify_current(self, module, *, max_source_operations=10_000, max_source_elements=64 << 20):
        if self not in find_zero_initializer_groups(
            module, max_source_operations=max_source_operations, max_source_elements=max_source_elements
        ):
            raise ValueError("zero initializer grouping differs from the complete current typed source")


def find_zero_initializer_groups(module, *, max_source_operations=10_000, max_source_elements=64 << 20):
    """Propose contiguous one-use zero splat and exact signed rank-two MAC groups.

    The source remains unchanged, including every scalar operation and result.
    A consumer must keep full source-op/task accounting while omitting virtual
    storage, and independently check its actual initializer/readout realization.
    Shared constants may stay in another task; shared or returned splats cannot
    disappear. Unknown scalar, layout, aliasing or arithmetic semantics retain
    the original route. No model identity or provenance label selects a group.
    """
    from merlin.transforms.dense_ir import recognize_signed_integer_matmul

    if type(module) is not builtin.ModuleOp:
        raise ValueError("zero initializer grouping requires the current original typed module")
    if any(type(value) is not int or value <= 0 for value in (max_source_operations, max_source_elements)):
        raise ValueError("zero initializer grouping requires explicit positive source budgets")
    functions = [op for op in module.body.block.ops if type(op) is func.FuncOp]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        return ()
    block = functions[0].body.block
    operations = tuple(block.ops)
    if len(operations) > max_source_operations:
        raise ValueError("zero initializer source operation budget exceeded")
    module.verify()
    if not operations or type(operations[-1]) is not func.ReturnOp:
        return ()
    if any(op.successors or type(op) is func.ReturnOp for op in operations[:-1]):
        return ()
    ordinals = {op: index for index, op in enumerate(operations[:-1])}
    groups, source_digest = [], None
    for ordinal, operation in enumerate(operations[:-1]):
        if not isinstance(operation, GenericOp):
            continue
        # Frontends may register a parser-only subclass of the standard generic
        # operation. Reconstruct its complete standard shell before applying
        # the strict recognizer; preserve all fields and unknown attributes.
        detached = GenericOp.create(
            operands=operation.operands,
            result_types=[value.type for value in operation.results],
            attributes=dict(operation.attributes),
            properties=dict(operation.properties),
            successors=operation.successors,
            regions=[region.clone() for region in operation.regions],
        )
        for inner in detached.walk():
            for key in tuple(inner.attributes):
                if key.startswith("prov."):
                    del inner.attributes[key]
        try:
            recognize_signed_integer_matmul(detached, ("i8", "i8", "i32"))
        except ValueError:
            continue
        finally:
            # Cloned operations temporarily add uses to their external SSA
            # operands. Restore those exact lists before testing single use.
            detached.drop_all_references()
        lhs, rhs, initializer = operation.operands
        shapes = [tuple(value.type.get_shape()) for value in (lhs, rhs, initializer)]
        if any(len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape) for shape in shapes):
            continue
        (m, k), (rhs_k, n), output = shapes
        if rhs_k != k or output != (m, n) or m * k + k * n + m * n > max_source_elements:
            continue
        splat = initializer.owner
        index = ordinals.get(splat)
        if (
            index != ordinal - 1
            or index is None
            or _name(splat) != "tensor.splat"
            or len(splat.operands) != 1
            or tuple(splat.results) != (initializer,)
            or splat.regions
            or splat.successors
            or not _provenance_only(splat)
            or len(tuple(initializer.uses)) != 1
            or next(iter(initializer.uses)).operation is not operation
        ):
            continue
        scalar = splat.operands[0]
        constant = scalar.owner
        if (
            type(constant) is not arith.ConstantOp
            or constant.parent is not block
            or ordinals.get(constant, ordinal) >= index
            or tuple(constant.results) != (scalar,)
            or type(constant.value) is not builtin.IntegerAttr
            or constant.value.value.data != 0
            or scalar.type != initializer.type.element_type
            or not _provenance_only(constant, properties=("value",))
        ):
            continue
        indices = (index, ordinal)
        virtual = ((index, 0),)
        if ordinals[constant] == index - 1 and len(tuple(scalar.uses)) == 1:
            indices = (index - 1, *indices)
            virtual = ((index - 1, 0), *virtual)
        if source_digest is None:
            source_digest = _digest(module)
        groups.append(ZeroInitializerGroup(
            source_digest, indices, ordinal, index, virtual, (m, k, n), m * n, m * n * 4
        ))
    return tuple(groups)
