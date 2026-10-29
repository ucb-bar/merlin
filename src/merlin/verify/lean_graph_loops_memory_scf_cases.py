"""Independent actual changed source families for the nested memory-SCF bridge."""

from __future__ import annotations

from dataclasses import dataclass

from .lean_graph_loops_reuse_cases import Case
from .lean_graph_loops_reuse_cases import source as epoch_source


@dataclass(frozen=True)
class CompositionCase:
    epoch: Case
    tile: int
    prefixes: bool = False


CASES = (
    CompositionCase(Case("rank_one_tail", (5,), 8, 1, 3, index_bits=32), 2),
    CompositionCase(Case("rank_two_tail", (5, 3), 16, 8, 3), 3),
    CompositionCase(Case("rank_two_one_row", (1, 7), 64, 64, 2), 4),
    CompositionCase(Case("rank_two_prefix", (4, 2), 32, 16, 2), 3, True),
    CompositionCase(Case("other_live_slot", (3, 2), 32, 16, 3, True), 2),
    CompositionCase(Case("compact_small", (3, 4), 32, 64, 3), 2),
    CompositionCase(Case("compact_huge", (1_000_000_000, 1024), 32, 64, 3), 2),
)


def source(case):
    text = epoch_source(case.epoch)
    if not case.prefixes:
        return text
    for epoch in range(case.epoch.epochs):
        line = f"   scf.for %iv{epoch}_1 = %zero to %bound1 step %one {{"
        prefix = f"   %row{epoch} = arith.addi %seed{epoch}, %seed{epoch} : i32\n"
        if line not in text:
            raise ValueError("rank2 outer initializer missing from independent source generator")
        text = text.replace(line, prefix + line)
        line = f"    memref.store %seed{epoch},"
        prefix = f"    %cell{epoch} = arith.addi %row{epoch}, %seed{epoch} : i32\n"
        if line not in text:
            raise ValueError("rank2 leaf store missing from independent source generator")
        text = text.replace(line, prefix + f"    memref.store %cell{epoch},")
    return text
