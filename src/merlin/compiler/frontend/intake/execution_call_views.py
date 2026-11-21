"""Independent view contracts for execution-owned model call expansion.

Reuse original static SSA coordinate recovery and standard tensor reassociation.
LLVM a47bddccec30255619bb8c37fa59700e661d4e66 TensorOps.cpp and tensor/ops.mlir
define element-count preservation and rank-zero reshapes. This is logical value
adaptation, never public physical alias or mutation permission.
"""

import math

from xdsl.dialects import builtin, tensor

from merlin.compiler.frontend.intake.derived_source_views import PROFILE as SOURCE_PROFILE
from merlin.compiler.frontend.intake.derived_source_views import checked_source_views
from merlin.compiler.frontend.intake.execution_runtime_views import POLICY as RUNTIME_POLICY
from merlin.compiler.frontend.intake.execution_runtime_views import PROFILE as RUNTIME_PROFILE
from merlin.compiler.frontend.intake.execution_source_coordinates import BINDING_PROFILE as COORDINATE_PROFILE
from merlin.compiler.frontend.intake.execution_source_coordinates import POLICY as SOURCE_POLICY
from merlin.compiler.frontend.intake.execution_source_coordinates import PROFILE as RUNTIME_SOURCE_PROFILE
from merlin.compiler.frontend.intake.execution_source_coordinates import SourceCoordinates

PROFILE = "t2m.execution-checked-call-views.v1"
POLICY = "dense-row-major-v1"
FIELDS = {"tensor_views", "source_tensor_views", "source_view_policy"}


def validate(bindings):
    records = bindings["calls"]
    present = any(FIELDS & set(record) for record in records)
    required = (
        COORDINATE_PROFILE
        if any("source_view_policy" in record for record in records)
        else RUNTIME_PROFILE
        if any(record.get("tensor_views") == RUNTIME_POLICY for record in records)
        else PROFILE
    )
    if present != ("execution_view_binding_profile" in bindings) or (
        present and bindings["execution_view_binding_profile"] != required
    ):
        raise ValueError("executed views require their explicit source-checked binding profile")
    for record in records:
        if "source_view_policy" in record and (
            record["source_view_policy"] != SOURCE_POLICY or not record.get("source_tensor_views")
        ):
            raise ValueError("runtime source coordinates require the explicit policy and actual source views")
        if "tensor_views" in record and record["tensor_views"] not in (POLICY, RUNTIME_POLICY):
            raise ValueError("executed tensor view has an unknown policy")
        if "source_tensor_views" in record and (
            not isinstance(record["source_tensor_views"], dict) or not record["source_tensor_views"]
        ):
            raise ValueError("executed source views require actual complete coordinate records")


def captured(record, attributes):
    return {
        name: attributes.data[schema].value.data
        for name, schema in record["runtime_scalar_arguments"].items()
        if isinstance(attributes.data.get(schema), builtin.IntegerAttr) and attributes.data[schema].type == builtin.i64
    }


def source_views(record, by_name, attributes, provenance, *, scalars=None, append=None):
    views = record.get("source_tensor_views")
    if views is not None:
        if record.get("source_view_policy") == SOURCE_POLICY:
            return SourceCoordinates(append).check_views(
                views, record["source_operand_bindings"], by_name, scalars, provenance
            )
        checked_source_views(
            views, record["source_operand_bindings"], by_name, captured(record, attributes), provenance
        )
    return views


def concrete(buffer, record, attributes, element):
    values = captured(record, attributes)
    shape = [values.get(n) if isinstance(n, str) else n for n in buffer["shape"]]
    if not all(type(n) is int and n >= 0 for n in shape):
        raise ValueError("executed tensor view needs source-bound concrete dimensions")
    return builtin.TensorType(element, shape)


def reshape(value, target, append, *, role):
    source = value.type
    if (
        not isinstance(source, builtin.TensorType)
        or not isinstance(target, builtin.TensorType)
        or not isinstance(source.encoding, builtin.NoneAttr)
        or not isinstance(target.encoding, builtin.NoneAttr)
        or source.element_type != target.element_type
        or any(n < 0 for n in (*source.get_shape(), *target.get_shape()))
        or math.prod(source.get_shape()) != math.prod(target.get_shape())
    ):
        raise ValueError("executed dense views require static equal element counts and exact element type")
    if source == target:
        return value
    flat = builtin.TensorType(target.element_type, [math.prod(target.get_shape())])

    def group(rank):
        return builtin.ArrayAttr(
            [builtin.ArrayAttr(builtin.IntegerAttr(i, builtin.i64) for i in range(rank))] if rank else []
        )

    rank = len(source.get_shape())
    if rank > 1:
        value = append(
            tensor.CollapseShapeOp.build(
                operands=[value], result_types=[flat], properties={"reassociation": group(rank)}
            ),
            role + "-collapse",
        )
    elif rank == 0:
        value = append(
            tensor.ExtractOp.build(operands=[value, []], result_types=[target.element_type]), role + "-scalar-extract"
        )
        value = append(tensor.FromElementsOp(value), role + "-scalar-pack")
    rank = len(target.get_shape())
    if rank > 1:
        value = append(tensor.ExpandShapeOp(value, [], group(rank), target.get_shape(), target), role + "-expand")
    elif rank == 0:
        value = append(
            tensor.CollapseShapeOp.build(
                operands=[value], result_types=[target], properties={"reassociation": group(0)}
            ),
            role + "-collapse",
        )
    return value


def mark(cloned, views, policy=None):
    if views:
        cloned.attributes["prov.kernel_source_view_profile"] = builtin.StringAttr(
            RUNTIME_SOURCE_PROFILE if policy == SOURCE_POLICY else SOURCE_PROFILE
        )
        cloned.attributes["prov.kernel_view_arguments"] = builtin.ArrayAttr(builtin.StringAttr(name) for name in views)
