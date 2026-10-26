"""Independent actual-source cases for shape/reuse/segmentation composition.

These are test inputs to pinned production APIs, never workload-name branches
in the compiler. Cell counts remain symbolic in the graph proof.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    rows: int
    columns: int
    tile: int
    index_bits: int = 64
    element_bits: int = 32


CASES = (
    Case("two_epochs", 2, 3, 2),
    Case("tail_nonunit", 5, 2, 3),
    Case("index32", 3, 4, 2, index_bits=32),
    Case("single_row", 1, 4, 2),
    Case("element16", 4, 3, 3, element_bits=16),
    Case("large_symbolic", 65536, 2, 8),
)


def source(case):
    if min(case.rows, case.columns) < 0 or case.element_bits not in (8, 16, 32, 64):
        raise ValueError("test case requires static dimensions and supported integer element width")
    scalar = "i" + str(case.element_bits)
    buffer = f"memref<{case.rows}x{case.columns}x{scalar}>"
    caller = f"memref<1x{scalar}>"
    lines = [
        "builtin.module {",
        f" func.func @kernel(%seed0: {scalar}, %seed1: {scalar}, %caller: {caller}) -> ({scalar}, index) {{",
        "  %zero = arith.constant 0 : index",
        "  %one = arith.constant 1 : index",
        f"  %bound0 = arith.constant {case.rows} : index",
        f"  %bound1 = arith.constant {case.columns} : index",
    ]
    for epoch in range(2):
        lines += [
            f"  %buffer{epoch} = memref.alloc() {{alignment = 16 : i64}} : {buffer}",
            f'  %view{epoch} = "memref.cast"(%buffer{epoch}) : ({buffer}) -> {buffer}',
            f'  %again{epoch} = "memref.cast"(%view{epoch}) : ({buffer}) -> {buffer}',
            f"  %rows{epoch} = memref.dim %again{epoch}, %zero : {buffer}",
            f"  %columns{epoch} = memref.dim %again{epoch}, %one : {buffer}",
            f"  scf.for %iv{epoch}_0 = %zero to %bound0 step %one {{",
            f"   scf.for %iv{epoch}_1 = %zero to %bound1 step %one {{",
            f"    memref.store %seed{epoch}, %again{epoch}[%iv{epoch}_0, %iv{epoch}_1] : {buffer}",
            "    scf.yield",
            "   }",
            "   scf.yield",
            "  }",
            f"  %first{epoch} = memref.load %again{epoch}[%zero, %zero] : {buffer}",
            f"  %updated{epoch} = arith.addi %first{epoch}, %seed{epoch} : {scalar}",
            f"  memref.store %updated{epoch}, %again{epoch}[%zero, %zero] : {buffer}",
            f"  %value{epoch} = memref.load %again{epoch}[%zero, %zero] : {buffer}",
            f"  %dim{epoch} = memref.dim %again{epoch}, %zero : {buffer}",
            f"  %observed{epoch} = func.call @touch(%value{epoch}, %caller) : ({scalar}, {caller}) -> {scalar}",
            f"  memref.dealloc %buffer{epoch} : {buffer}",
        ]
    lines += [
        f"  %result = arith.addi %observed0, %observed1 : {scalar}",
        f"  func.return %result, %dim1 : {scalar}, index",
        " }",
        f" func.func @touch(%value: {scalar}, %caller: {caller}) -> {scalar} {{",
        "  %zero = arith.constant 0 : index",
        f"  %old = memref.load %caller[%zero] : {caller}",
        f"  %new = arith.addi %old, %value : {scalar}",
        f"  memref.store %new, %caller[%zero] : {caller}",
        f"  func.return %new : {scalar}",
        " }",
        " func.func @untouched(%input: i32) -> i32 {",
        "  %one = arith.constant 1 : i32",
        "  %result = arith.addi %input, %one : i32",
        "  func.return %result : i32",
        " }",
        "}",
    ]
    return "\n".join(lines) + "\n"
