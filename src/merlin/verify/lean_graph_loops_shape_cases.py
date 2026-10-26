"""Independent whole shape-normalization source families and memory-chain input."""

from dataclasses import dataclass

from .lean_graph_loops_reuse_cases import Case as EpochCase
from .lean_graph_loops_reuse_cases import source as epoch_source


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    bits: int
    axis: int
    index_bits: int = 64
    nested: bool = False
    callee: bool = False


CASES = (
    Case("rank_one", (7,), 8, 0, 32),
    Case("rank_two", (2, 3), 32, 1),
    Case("rank_three", (3, 5, 2), 64, 2),
    Case("nested_captures", (4, 3), 16, 0, nested=True),
    Case("zero_metadata_only", (0, 5), 32, 0),
    Case("compact_huge", (1_000_000_000, 1024), 32, 1),
)


def source(case):
    scalar = f"i{case.bits}"
    kind = "memref<" + "x".join(map(str, case.shape)) + "x" + scalar + ">"
    sink = f"memref<1x{scalar}>"
    zero = ", ".join("%zero" for _ in case.shape)
    other = (case.axis + 1) % len(case.shape)
    lines = [
        "builtin.module {",
        f" func.func @probe(%a: {kind}, %b: {kind}, %sink: {sink}) -> (index, index, {scalar}) {{",
        "  %zero = arith.constant 0 : index",
        "  %one = arith.constant 1 : index",
        f"  %axis = arith.constant {case.axis} : index",
        f"  %other = arith.constant {other} : index",
        f'  %first = "memref.cast"(%a) : ({kind}) -> {kind}',
        f'  %second = "memref.cast"(%first) : ({kind}) -> {kind}',
        f"  %d0 = memref.dim %second, %axis : {kind}",
        f"  %d1 = memref.dim %b, %other : {kind}",
    ]
    if case.nested:
        lines.extend(
            [
                "  scf.for %i = %zero to %one step %one {",
                f'   %local = "memref.cast"(%second) : ({kind}) -> {kind}',
                f"   %localdim = memref.dim %local, %axis : {kind}",
                f"   %inside = memref.load %local[{zero}] : {kind}",
                f"   memref.store %inside, %sink[%zero] : {sink}",
                "   scf.yield",
                "  }",
            ]
        )
    if 0 in case.shape:
        lines.append(f"  %value = memref.load %sink[%zero] : {sink}")
    else:
        lines.append(f"  %value = memref.load %second[{zero}] : {kind}")
        lines.append(f"  memref.store %value, %sink[%zero] : {sink}")
    if case.callee:
        lines = [
            f"  %d0 = func.call @dimension(%second) : ({kind}) -> index"
            if line.startswith("  %d0 = memref.dim")
            else line
            for line in lines
        ]
    lines.extend([f"  func.return %d0, %d1, %value : index, index, {scalar}", " }"])
    if case.callee:
        lines.extend(
            [
                f" func.func @dimension(%arg: {kind}) -> index {{",
                f"  %axis = arith.constant {case.axis} : index",
                f'  %cast = "memref.cast"(%arg) : ({kind}) -> {kind}',
                f"  %dim = memref.dim %cast, %axis : {kind}",
                "  func.return %dim : index",
                " }",
            ]
        )
    lines.extend([" func.func @untouched(%value: i32) -> i32 {", "  func.return %value : i32", " }", "}"])
    return "\n".join(lines) + "\n"


def memory_chain_source():
    text = epoch_source(EpochCase("shape_memory_chain", (2, 3), 32, 16, 2))
    text = text.replace("memref<?xi32>", "memref<1xi32>")
    lines = []
    for line in text.splitlines():
        allocated = None
        for epoch in (0, 1):
            if line.startswith(f"  %buffer{epoch} = memref.alloc"):
                allocated = epoch
                break
        if allocated is not None:
            epoch = allocated
            lines.extend(
                [
                    line,
                    f'  %view{epoch} = "memref.cast"(%buffer{epoch}) : (memref<2x3xi32>) -> memref<2x3xi32>',
                    f'  %again{epoch} = "memref.cast"(%view{epoch}) : (memref<2x3xi32>) -> memref<2x3xi32>',
                    f"  %rows{epoch} = memref.dim %again{epoch}, %zero : memref<2x3xi32>",
                    f"  %columns{epoch} = memref.dim %again{epoch}, %one : memref<2x3xi32>",
                ]
            )
        else:
            for epoch in (0, 1):
                if not line.lstrip().startswith("memref.dealloc "):
                    line = line.replace(f"%buffer{epoch}", f"%again{epoch}")
            lines.append(line)
    return "\n".join(lines) + "\n"
