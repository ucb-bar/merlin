"""Candidate scheduling proposals for independently checked staged effects.

These algorithms do not establish target applicability or emitted-code
qualification. Call them through ``rewrite_staged_dense`` so every proposal
passes the fixed complete-product and resource checks.
"""

from __future__ import annotations

from .staged_dense import AccumulatorReadout, AccumulatorZero, CompletionFence


def order_output_tiles_n_before_m(operations, problem, contract):
    """Move complete output tiles to column-major traversal without new storage.

    Each movable tile includes its zero, full reduction, readout and following
    completion fence. No operation inside a tile is reordered, and no fence is
    removed. A later RHS-residency rewrite can use the resulting panel locality.
    If earlier rewrites removed a needed load, the independent checker must
    reject the resulting proposal; pass order is therefore observable.
    """
    del problem
    if not contract.external_buffers_disjoint or not contract.rhs_immutable:
        return operations
    if type(operations) is not tuple:
        raise ValueError("output-tile ordering requires immutable staged effects")
    cursor = 0
    while cursor < len(operations) and type(operations[cursor]) is CompletionFence:
        cursor += 1
    prefix = operations[:cursor]
    blocks = []
    while cursor < len(operations):
        if type(operations[cursor]) is CompletionFence:
            if not all(type(op) is CompletionFence for op in operations[cursor:]):
                raise ValueError("output-tile ordering cannot detach an inter-tile effect")
            break
        start = cursor
        if type(operations[cursor]) is not AccumulatorZero:
            raise ValueError("output-tile ordering requires complete independently zeroed tiles")
        cursor += 1
        while cursor < len(operations) and type(operations[cursor]) is not AccumulatorReadout:
            if type(operations[cursor]) is AccumulatorZero:
                raise ValueError("output-tile ordering cannot split overlapping accumulator lifetimes")
            cursor += 1
        if cursor + 1 >= len(operations) or type(operations[cursor + 1]) is not CompletionFence:
            raise ValueError("output-tile ordering requires complete readout and completion fence")
        readout = operations[cursor]
        cursor += 2
        blocks.append(((readout.output_col, readout.output_row), operations[start:cursor]))
    keys = [key for key, _ in blocks]
    if len(keys) != len(set(keys)):
        raise ValueError("output-tile ordering cannot duplicate a logical output tile")
    ordered = sorted(blocks, key=lambda block: block[0])
    return (*prefix, *(op for _, block in ordered for op in block), *operations[cursor:])
