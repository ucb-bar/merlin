"""Independent complete source programs for actual buffer-reuse qualification."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    bits: int
    alignment: int
    epochs: int
    overlap: bool = False
    index_bits: int = 64

    def record(self):
        return asdict(self)


CASES = (
    Case("two_epochs", (2, 3), 32, 16, 2),
    Case("three_epochs", (3, 2), 16, 8, 3),
    Case("one_dim", (5,), 8, 1, 3, index_bits=32),
    Case("three_dim", (2, 3, 2), 64, 64, 2),
    Case("live_other_slot", (2, 2), 32, 16, 3, True),
    Case("large", (1_000_000_000, 1024), 32, 64, 3),
    Case("large_shape", (3, 4), 32, 64, 3),
)


def source(case: Case):
    scalar = f"i{case.bits}"
    kind = "memref<" + "x".join(str(value) for value in case.shape) + "x" + scalar + ">"
    caller = f"memref<?x{scalar}>"
    args = [f"%seed{i}: {scalar}" for i in range(case.epochs)] + [f"%caller: {caller}"]
    lines = [
        "builtin.module {",
        f" func.func @kernel({', '.join(args)}) -> ({scalar}, index) {{",
        "  %zero = arith.constant 0 : index",
        "  %one = arith.constant 1 : index",
    ]
    for axis, extent in enumerate(case.shape):
        lines.append(f"  %bound{axis} = arith.constant {extent} : index")
    indices = ", ".join("%zero" for _ in case.shape)
    for epoch in range(case.epochs):
        lines.append(f"  %buffer{epoch} = memref.alloc() {{alignment = {case.alignment} : i64}} : {kind}")
        indent = "  "
        for axis in range(len(case.shape)):
            lines.append(indent + f"scf.for %iv{epoch}_{axis} = %zero to %bound{axis} step %one {{")
            indent += " "
        coordinate = ", ".join(f"%iv{epoch}_{axis}" for axis in range(len(case.shape)))
        lines.append(indent + f"memref.store %seed{epoch}, %buffer{epoch}[{coordinate}] : {kind}")
        for _axis in case.shape:
            lines.append(indent + "scf.yield")
            indent = indent[:-1]
            lines.append(indent + "}")
        lines.extend(
            [
                f"  %first{epoch} = memref.load %buffer{epoch}[{indices}] : {kind}",
                f"  %updated{epoch} = arith.addi %first{epoch}, %seed{epoch} : {scalar}",
                f"  memref.store %updated{epoch}, %buffer{epoch}[{indices}] : {kind}",
                f"  %value{epoch} = memref.load %buffer{epoch}[{indices}] : {kind}",
                f"  %dim{epoch} = memref.dim %buffer{epoch}, %zero : {kind}",
                f"  %observed{epoch} = func.call @touch(%value{epoch}, %caller) : ({scalar}, {caller}) -> {scalar}",
            ]
        )
        if case.overlap and epoch == 0:
            continue
        if case.overlap and epoch == 1:
            lines.extend(
                [
                    f"  %late = memref.load %buffer0[{indices}] : {kind}",
                    f"  memref.store %late, %caller[%zero] : {caller}",
                    f"  memref.dealloc %buffer0 : {kind}",
                ]
            )
        lines.append(f"  memref.dealloc %buffer{epoch} : {kind}")
    lines.extend(
        [
            f"  %result = arith.addi %observed0, %observed{case.epochs - 1} : {scalar}",
            f"  func.return %result, %dim{case.epochs - 1} : {scalar}, index",
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
    )
    return "\n".join(lines) + "\n"


def resource_gap_source():
    """Source peaks at8 bytes; retained reuse plus a differently shaped8-byte
    object peaks at16. This is an explicit total-allocator premise countermodel.
    """
    lines = [
        "builtin.module {",
        " func.func @kernel(%seed0: i32, %seed1: i32, %seed2: i32, %caller: memref<?xi32>) -> (i32, index) {",
        "  %zero = arith.constant 0 : index",
        "  %one = arith.constant 1 : index",
        "  %two = arith.constant 2 : index",
    ]
    for epoch, shape in enumerate(((2,), (1, 2), (2,))):
        kind = "memref<" + "x".join(str(value) for value in shape) + "xi32>"
        lines.append(f"  %buffer{epoch} = memref.alloc() {{alignment = 8 : i64}} : {kind}")
        indent = "  "
        for axis, extent in enumerate(shape):
            bound = "%two" if extent == 2 else "%one"
            lines.append(indent + f"scf.for %iv{epoch}_{axis} = %zero to {bound} step %one {{")
            indent += " "
        coordinates = ", ".join(f"%iv{epoch}_{axis}" for axis in range(len(shape)))
        lines.append(indent + f"memref.store %seed{epoch}, %buffer{epoch}[{coordinates}] : {kind}")
        for _axis in shape:
            lines.append(indent + "scf.yield")
            indent = indent[:-1]
            lines.append(indent + "}")
        indices = ", ".join("%zero" for _axis in shape)
        lines.extend(
            [
                f"  %first{epoch} = memref.load %buffer{epoch}[{indices}] : {kind}",
                f"  %updated{epoch} = arith.addi %first{epoch}, %seed{epoch} : i32",
                f"  memref.store %updated{epoch}, %buffer{epoch}[{indices}] : {kind}",
                f"  %value{epoch} = memref.load %buffer{epoch}[{indices}] : {kind}",
                f"  %dim{epoch} = memref.dim %buffer{epoch}, %zero : {kind}",
                f"  %observed{epoch} = func.call @touch(%value{epoch}, %caller) : (i32, memref<?xi32>) -> i32",
                f"  memref.dealloc %buffer{epoch} : {kind}",
            ]
        )
    lines.extend(
        [
            "  %result = arith.addi %observed0, %observed2 : i32",
            "  func.return %result, %dim2 : i32, index",
            " }",
            " func.func @touch(%value: i32, %caller: memref<?xi32>) -> i32 {",
            "  %zero = arith.constant 0 : index",
            "  %old = memref.load %caller[%zero] : memref<?xi32>",
            "  %new = arith.addi %old, %value : i32",
            "  memref.store %new, %caller[%zero] : memref<?xi32>",
            "  func.return %new : i32",
            " }",
            " func.func @untouched(%input: i32) -> i32 {",
            "  %one = arith.constant 1 : i32",
            "  %result = arith.addi %input, %one : i32",
            "  func.return %result : i32",
            " }",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def inputs():
    return [(case.record(), source(case)) for case in CASES] + [
        (
            dict(
                name="resource_gap",
                shape=(2,),
                bits=32,
                alignment=8,
                epochs=3,
                overlap=False,
                index_bits=64,
                object_shapes=((2,), (1, 2), (2,)),
                object_bits=(32, 32, 32),
                resource_exhaustion_countermodel=True,
            ),
            resource_gap_source(),
        )
    ]
