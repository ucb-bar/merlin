"""Typed transport/residency boundaries for an original sliding input window.

These operations encode no target instruction or timing law. A binding must
implement complete transport and complete consumer use at their explicit
boundaries; source content and target effects remain separate obligations.
"""

from xdsl.dialects.builtin import IntegerAttr, i1, i64
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import IRDLOperation, irdl_attr_definition, irdl_op_definition, operand_def, prop_def, result_def


@irdl_attr_definition
class TransferType(ParametrizedAttribute, TypeAttribute):
    name = "rolling_window.transfer"


@irdl_attr_definition
class ResidentType(ParametrizedAttribute, TypeAttribute):
    name = "rolling_window.resident"


@irdl_attr_definition
class ConsumerType(ParametrizedAttribute, TypeAttribute):
    name = "rolling_window.consumer"


@irdl_op_definition
class BeginOp(IRDLOperation):
    name = "rolling_window.begin"
    transfer = result_def(TransferType)


@irdl_op_definition
class StageRowOp(IRDLOperation):
    """Copy row_bytes when valid; otherwise write exactly those zero bytes."""

    name = "rolling_window.stage_row"
    transfer = operand_def(TransferType)
    image = operand_def(i64)
    input_row = operand_def(i64)
    source_byte = operand_def(i64)
    cache_byte = operand_def(i64)
    valid = operand_def(i1)
    row_bytes = prop_def(IntegerAttr)


@irdl_op_definition
class AwaitOp(IRDLOperation):
    name = "rolling_window.await"
    transfer = operand_def(TransferType)
    resident = result_def(ResidentType)


@irdl_op_definition
class ConsumeOp(IRDLOperation):
    """The original consumer may read only its checked original window view."""

    name = "rolling_window.consume"
    resident = operand_def(ResidentType)
    image = operand_def(i64)
    output_y = operand_def(i64)
    consumer = result_def(ConsumerType)


@irdl_op_definition
class CompleteOp(IRDLOperation):
    name = "rolling_window.complete"
    consumer = operand_def(ConsumerType)


RollingWindow = Dialect(
    "rolling_window",
    [BeginOp, StageRowOp, AwaitOp, ConsumeOp, CompleteOp],
    [TransferType, ResidentType, ConsumerType],
)
