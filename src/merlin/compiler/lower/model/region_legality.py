"""Which consecutive device groups of a captured model may be answered as ONE region kernel.

Today the whole-program statement (:mod:`.whole_program`) puts every device group to a package one
at a time: a package answers a matmul, or a residual add, and never both together, so it cannot fuse
a residual add onto the contraction that feeds it or keep an intermediate on the accelerator between
two groups. This module answers the one question that has to be answered BEFORE any of that is
offered: which runs of groups even COULD be one region, from the statement's own dataflow alone.

**The rule, and why it is the builder's to apply, not the package's.** A run of consecutive groups
``g_i, g_{i+1}, ..., g_j`` may be asked as one region exactly when every internal boundary in it is a
SINGLE-CONSUMER INTERMEDIATE THAT IS NOT A MODEL OUTPUT: group ``g_k``'s committed tensor (``k`` in
``(i, j]``... rather ``[i, j)``) is read by no group other than ``g_{k+1}``, and it is not the
model's own result. Both halves matter. A tensor two groups both read cannot be folded into one
kernel's private intermediate -- the OTHER reader still needs it as a named buffer. A tensor that IS
the model's result cannot disappear into a region either: something has to read it out, and the
statement already names the last device group's product as that result
(:func:`merlin.compiler.lower.model.whole_program.whole_program_buffer`'s own convention, restated here rather
than re-derived, so the two cannot disagree about which group produces the output).

The rule is a fact about the model's dataflow, never about a name, a shape, an op spelling, or a
group count -- a two-layer synthetic model and a seventy-one-group capture are found by the same
walk, and adding a second target requires no change here at all.

**What legality does NOT decide.** A legal window is an OFFER, not a fusion: :mod:`.region_capsule`
still has to ask a package for it, the package may decline (or answer group by group, which reads as
declining), and a region that the package does not answer falls back to asking its members one at a
time -- exactly as an unaccepted single-group ask always has.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["legal_regions"]


def _sink(group: Any) -> Any | None:
    """The one tensor a device group commits, as the whole-program statement reads it: the last
    member's first result. ``None`` for a member sequence with no result at all (a residency
    backend's own ``EVICT`` tail), which cannot be anyone's producer."""
    if not group.members or not group.members[-1].results:
        return None
    return group.members[-1].results[0]


def legal_regions(groups: Sequence[Any]) -> list[tuple[int, int]]:
    """Maximal runs of two or more consecutive device groups this program may put to a package as
    ONE region, as ``(first group index, last group index)`` pairs in ascending, non-overlapping
    order.

    Only DEVICE groups are considered (a group's placement is checked, never assumed); the one host
    region a closed model keeps is not a candidate for anything here. A model whose groups admit no
    legal boundary at all -- every intermediate is read twice, or every group but the last is a
    fan-out -- returns an empty list, and every group is still asked on its own, exactly as before
    this existed.
    """
    from merlin.compiler.dialects.lowering import compute_groups as CG

    device = [g for g in groups if g.placement != CG.HOST]
    if len(device) < 2:
        return []
    sinks: dict[int, Any] = {g.index: _sink(g) for g in device}
    consumers: dict[int, set[int]] = {index: set() for index, value in sinks.items() if value is not None}
    for group in device:
        for member in group.members:
            for operand in member.operands:
                for index, value in sinks.items():
                    if index != group.index and value is not None and operand is value:
                        consumers[index].add(group.index)
    order = [g.index for g in device]
    # THE MODEL'S OWN RESULT is the last device group's product -- the same convention
    # `whole_program_buffer` uses to declare the buffer's output, restated rather than re-derived so
    # the two cannot silently disagree about which group may not disappear into a region.
    result_group = order[-1]
    regions: list[tuple[int, int]] = []
    start = order[0]
    for position in range(len(order) - 1):
        this_index, next_index = order[position], order[position + 1]
        legal_edge = this_index in consumers and consumers[this_index] == {next_index} and this_index != result_group
        if not legal_edge:
            if this_index != start:
                regions.append((start, this_index))
            start = next_index
    if order[-1] != start:
        regions.append((start, order[-1]))
    return regions
