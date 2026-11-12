"""Explicit scheduling permission for independent source FMA chains.

The selected implementation keeps each lane's multiply, addend and single
rounding. It may schedule independent lanes together under the supplied source
effects contract. CPU instructions and register constraints belong to explicitly
selected host implementations, shared in Merlin independently of accelerators.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFmaBatchContract:
    lanes: int
    fused_single_rounding: bool
    operand_order_preserved: bool
    finite_operands_and_results: bool
    private_disjoint_storage: bool
    stable_rounding: bool
    gradual_underflow: bool
    nontrapping: bool
    exception_flags_unobserved: bool
    errno_unobserved: bool

    def validate(self) -> None:
        if type(self.lanes) is not int or self.lanes != 8:
            raise ValueError("explicit eight independent source FMA lanes required")
        if any(type(value) is not bool or not value for key, value in self.__dict__.items() if key != "lanes"):
            raise ValueError("complete source arithmetic and effects contract required")


def emit_source_fma_batch_permission(contract: SourceFmaBatchContract) -> str:
    """Select an implementation for separately proved finite private chains."""
    if not isinstance(contract, SourceFmaBatchContract):
        raise ValueError("typed independent source FMA batch contract required")
    contract.validate()
    return (
        "#define MERLIN_ENABLE_SOURCE_FMA_BATCH_8 1\n"
        "#if !defined(MERLIN_SOURCE_F32_FMA_EIGHT)\n"
        '#error "explicit independent source FMA batch provider required"\n'
        "#endif\n"
    )
