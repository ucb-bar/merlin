"""Portable scheduling decision IR shared with standalone target compilers."""

from xdsl.dialects.builtin import DenseArrayBase, IntegerAttr, i64
from xdsl.irdl import IRDLOperation, attr_def, irdl_op_definition, traits_def
from xdsl.traits import NoMemoryEffect
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class DenseScheduleOp(IRDLOperation):
    """A selected contraction block, with explicit byte and command bounds.

    This records a compile-time choice and executes no command. The target
    adapter verifies its realization before emitting this decision.
    """

    name = "schedule.block_contraction"
    problem = attr_def(DenseArrayBase)
    primitive_tile = attr_def(DenseArrayBase)
    block_tiles = attr_def(DenseArrayBase)
    element_bytes = attr_def(DenseArrayBase)
    storage_bytes = attr_def(DenseArrayBase)
    storage_capacity = attr_def(DenseArrayBase)
    static_commands = attr_def(IntegerAttr)
    command_budget = attr_def(IntegerAttr)
    traits = traits_def(NoMemoryEffect())

    def verify_(self):
        values = {}
        for name, count in (
            ("problem", 3),
            ("primitive_tile", 3),
            ("block_tiles", 3),
            ("element_bytes", 3),
            ("storage_bytes", 2),
            ("storage_capacity", 2),
        ):
            value = getattr(self, name)
            data = tuple(value.get_values())
            if value.elt_type != i64 or len(data) != count or any(v <= 0 for v in data):
                raise VerifyException(f"{name} requires {count} positive i64 values")
            values[name] = data
        tm, tk, tn = values["primitive_tile"]
        bm, bk, bn = values["block_tiles"]
        a, b, c = values["element_bytes"]
        expected = (bk * (bm * tm * tk * a + bn * tk * tn * b), bm * bn * tm * tn * c)
        if expected != values["storage_bytes"]:
            raise VerifyException("dense schedule storage differs from its tile representation")
        if any(used > capacity for used, capacity in zip(expected, values["storage_capacity"])):
            raise VerifyException("dense schedule exceeds available storage")
        sites, budget = self.static_commands, self.command_budget
        if sites.type != i64 or budget.type != i64 or not 0 <= sites.value.data <= budget.value.data:
            raise VerifyException("dense schedule exceeds its static command budget")


def dense_schedule_op(problem, block, resources, static_commands, budget):
    """Materialize a checked decision without importing a particular planner copy."""
    arrays = {
        "problem": (problem.m, problem.k, problem.n),
        "primitive_tile": (problem.tile_m, problem.tile_k, problem.tile_n),
        "block_tiles": (block.m, block.k, block.n),
        "element_bytes": (problem.lhs_bytes, problem.rhs_bytes, problem.accumulator_bytes),
        "storage_bytes": (block.operand_bytes, block.accumulator_bytes),
        "storage_capacity": (resources.operand_bytes, resources.accumulator_bytes),
    }
    op = DenseScheduleOp.create(
        attributes={
            **{name: DenseArrayBase.from_list(i64, values) for name, values in arrays.items()},
            "static_commands": IntegerAttr(static_commands, i64),
            "command_budget": IntegerAttr(budget, i64),
        }
    )
    op.verify()
    return op
