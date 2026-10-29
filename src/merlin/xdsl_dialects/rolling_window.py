"""Typed transport/residency boundaries for an original sliding input window.

These operations encode no target instruction or timing law. A binding must
implement complete transport and complete consumer use at their explicit
boundaries; source content and target effects remain separate obligations.
"""

from xdsl.dialects.builtin import IntegerAttr, i1, i64
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                       operand_def, prop_def, result_def, opt_region_def, traits_def)
from xdsl.traits import NoTerminator


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
class ConsumeGroupOp(IRDLOperation):
    """Consume an explicit contiguous interval of original output-row views.

    The complete source-row union must remain resident through this consumer's
    completion. This operation grants no arithmetic reordering or target effect.
    """

    name = "rolling_window.consume_group"
    resident = operand_def(ResidentType)
    image = operand_def(i64)
    output_y = operand_def(i64)
    output_rows = prop_def(IntegerAttr)
    consumer = result_def(ConsumerType)


@irdl_op_definition
class CompleteOp(IRDLOperation):
    name = "rolling_window.complete"
    consumer = operand_def(ConsumerType)


@irdl_op_definition
class StartConsumerOp(IRDLOperation):
    """Initialize one original grouped consumer with live retained state."""

    name = "rolling_window.start_consumer"
    resident = operand_def(ResidentType)
    image = operand_def(i64)
    output_y = operand_def(i64)
    output_rows = prop_def(IntegerAttr)
    consumer = result_def(ConsumerType)


@irdl_op_definition
class AdvanceConsumerOp(IRDLOperation):
    """Run a contiguous original reduction interval and retain its continuation.

    A run contains the whole stationary-weight fresh/stay traversal. It is an
    issue boundary, not completion; only the final real Complete ends use.
    """

    name = "rolling_window.advance_consumer"
    previous = operand_def(ConsumerType)
    first_run = prop_def(IntegerAttr)
    run_count = prop_def(IntegerAttr)
    consumer = result_def(ConsumerType)
    after_run = opt_region_def("single_block")
    traits = traits_def(NoTerminator())


@irdl_op_definition
class BeginRhsOp(IRDLOperation):
    """Bind the real assigned RHS rows; completion is the current full drain."""

    name = "rolling_window.begin_rhs"
    rhs_row = prop_def(IntegerAttr)
    transfer = result_def(TransferType)


@irdl_op_definition
class StageRhsOp(IRDLOperation):
    """Issue the next original RHS tile at the current panel's last use."""

    name = "rolling_window.stage_rhs"
    transfer = operand_def(TransferType)
    reduction_row = operand_def(i64)
    output_column = operand_def(i64)
    local_row = operand_def(i64)
    rows = operand_def(i64)
    columns = prop_def(IntegerAttr)


@irdl_op_definition
class FinishConsumerOp(IRDLOperation):
    """Read out only after every original reduction run has been issued."""

    name = "rolling_window.finish_consumer"
    previous = operand_def(ConsumerType)
    consumer = result_def(ConsumerType)


RollingWindow = Dialect(
    "rolling_window",
    [BeginOp, StageRowOp, AwaitOp, ConsumeOp, ConsumeGroupOp, CompleteOp,
     StartConsumerOp, AdvanceConsumerOp, FinishConsumerOp, BeginRhsOp, StageRhsOp],
    [TransferType, ResidentType, ConsumerType],
)
