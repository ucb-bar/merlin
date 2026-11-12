"""When is a model CLOSED -- that is, when is "its compute groups" the whole program?

One definition, in one place, because two spellings of this would drift and the second one would
silently accept a different set of models. The rule:

    A model is closed when its only host region is the quantization of a model ARGUMENT onto the
    integer grid.

That one region is the exception, and it is an exception on purpose rather than an oversight in the
command vocabulary. The caller hands the model a float image; something has to put it on the grid,
and that conversion is the program's INPUT DOMAIN rather than a step of it -- a device program
legitimately begins with its operands already quantized. Any OTHER host region computes BETWEEN the
groups, and then a program built from the groups is not the model: it would run a third of the layers
and report a number under the model's name.

Used by the group-model harness (which builds a bare-metal program of exactly these groups) and by
:mod:`merlin.compiler.lower.model.whole_program` (which states them as one command buffer). Both refuse on the
same predicate, so "closed" means the same thing in a measured cycle count and in an emitted buffer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["open_host_regions", "quantizes_an_input", "require_closed"]


def quantizes_an_input(group) -> bool:
    """The one host region a closed model keeps: a model ARGUMENT put on the integer grid.

    Both halves are load-bearing. ``stages == [QUANTIZE]`` means the region does nothing but the
    conversion -- a region that also computed would be work this rule would wave through. And the
    quantize's source being a BLOCK ARGUMENT means it converts something the caller handed in, not an
    intermediate: a requantize in the middle of a model is the same operation on a value the model
    itself produced, and leaving that on the host is exactly the split this predicate exists to catch.
    """
    from xdsl.ir import BlockArgument

    from merlin.compiler.dialects.lowering import compute_groups as CG

    if list(group.stages) != [CG.QUANTIZE] or not group.members:
        return False
    operands = list(getattr(group.members[-1], "operands", ()))
    return bool(operands) and isinstance(operands[0], BlockArgument)


def open_host_regions(groups: Sequence[Any]) -> list[Any]:
    """Every host region of ``groups`` that is NOT the permitted input quantization."""
    from merlin.compiler.dialects.lowering import compute_groups as CG

    return [g for g in groups if g.placement == CG.HOST and not quantizes_an_input(g)]


def require_closed(groups: Sequence[Any], *, error: type[Exception] = ValueError) -> None:
    """Raise ``error`` naming the host regions that keep this model from being closed.

    Named, never counted: "3 host regions" tells a reader nothing they can act on, while "group 12
    ['elementwise'] (input dtype 'fp32' not in elementwise_map formats ['int8'])" names the operation
    and the reason the target refused it.
    """
    open_host = open_host_regions(groups)
    if open_host:
        raise error(
            f"{len(open_host)} host region(s) compute between the groups, so a program of groups is not "
            f"the model: " + "; ".join(f"group {g.index} {list(g.stages)[:4]} ({g.reason})" for g in open_host[:4])
        )
