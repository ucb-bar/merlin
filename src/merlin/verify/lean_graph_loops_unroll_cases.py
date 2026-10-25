"""Independent actual-producer inputs for ordered unroll qualification.

These are source programs, never manufactured expected after-programs. Actual
4499 production in a subprocess supplies every changed candidate.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Case:
    name: str
    lower: int
    upper: int
    step: int
    factor: int
    carries: int
    bits: int
    effects: bool
    yield_mode: str = "updated"

    def record(self):
        return asdict(self)


CASES = (
    Case("inline_empty", 7, 7, 2, 5, 2, 32, True),
    Case("inline_single", 3, 4, 7, 4, 0, 64, True),
    Case("inline_many", 1, 8, 2, 7, 3, 64, True),
    Case("bulk_exact", 3, 27, 2, 3, 2, 64, True),
    Case("bulk_tail", 3, 25, 2, 3, 2, 64, True),
    Case("unit_stride", 0, 17, 1, 4, 1, 32, False),
    Case("no_carry", 2, 19, 3, 2, 0, 32, True),
    Case("forward_alias", 5, 12, 2, 9, 2, 64, True, "capture"),
    Case("forward_only", 0, 14, 2, 3, 3, 64, False, "identity"),
    Case("bulk_past_upper", 3, 37, 4, 3, 1, 32, False),
    Case("large", 3, 1_000_000_000_003, 7, 3, 2, 64, False),
    Case("large_shape", 3, 110, 7, 3, 2, 64, False),
)


def source(case: Case):
    types = ["i32"] * case.carries
    arguments = [f"%input{i}: i32" for i in range(case.carries)] + ["%capture: i32"]
    if case.effects:
        arguments.append("%memory: memref<?xi32>")
    result_types = types or ["i32"]
    lines = [
        "builtin.module {",
        f" func.func @kernel({', '.join(arguments)}) -> ({', '.join(result_types)}) {{",
        f"  %lower = arith.constant {case.lower} : index",
        f"  %upper = arith.constant {case.upper} : index",
        f"  %step = arith.constant {case.step} : index",
    ]
    if case.carries:
        bind = "%result = " if case.carries == 1 else f"%result:{case.carries} = "
        init = ", ".join(f"%carry{i} = %input{i}" for i in range(case.carries))
        suffix = f" iter_args({init}) -> ({', '.join(types)})"
    else:
        bind, suffix = "", ""
    lines.append(f"  {bind}scf.for %iv = %lower to %upper step %step{suffix} {{")
    yields = []
    if case.yield_mode != "identity":
        previous = "%capture"
        for i in range(case.carries):
            lines.append(f"   %next{i} = arith.addi %carry{i}, {previous} : i32")
            previous = f"%next{i}"
            yields.append(previous)
        if not case.carries and case.effects:
            lines.append("   %stored = arith.addi %capture, %capture : i32")
        if case.effects:
            stored = "%next0" if case.carries else "%stored"
            lines.append(f"   memref.store {stored}, %memory[%iv] : memref<?xi32>")
    if case.yield_mode == "identity":
        yields = [f"%carry{i}" for i in range(case.carries)]
    elif case.yield_mode == "capture":
        yields = ["%capture"] * case.carries
    lines.append("   scf.yield" + (" " + ", ".join(yields) + " : " + ", ".join(types) if yields else ""))
    lines.append("  }")
    result = ["%result" if case.carries == 1 else f"%result#{i}" for i in range(case.carries)]
    if result:
        lines.append(f"  %observer = arith.addi {result[0]}, %capture : i32")
        result[0] = "%observer"
    else:
        result = ["%capture"]
    lines.extend(
        [
            f"  func.return {', '.join(result)} : {', '.join(result_types)}",
            " }",
            " func.func @untouched(%value: i32) -> i32 {",
            "  %one = arith.constant 1 : i32",
            "  %result = arith.addi %value, %one : i32",
            "  func.return %result : i32",
            " }",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"
