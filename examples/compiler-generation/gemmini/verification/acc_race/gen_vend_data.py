"""Write vend_data.h: two initialised int8 operands for vend_repro.c.

    gen_vend_data.py <out.h> <rows> [cols]

Operand A in [1, 40], operand B in [41, 80] (so a+b, a and b are always distinguishable), from a fixed
seed so every build of a given shape carries the same bytes.
"""

import random
import sys


def main(out: str, rows: int, cols: int = 512) -> None:
    rng = random.Random(12345)
    n = rows * cols

    def arr(name: str, lo: int, hi: int) -> str:
        vals = ",".join(str(rng.randint(lo, hi)) for _ in range(n))
        return f"static const elem_t {name}[{n}] row_align(1) = {{{vals}}};\n"

    with open(out, "w") as fh:
        fh.write(f"#define VEND_ROWS {rows}\n#define VEND_COLS {cols}\n")
        fh.write(arr("VA", 1, 40))
        fh.write(arr("VB", 41, 80))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), *(int(x) for x in sys.argv[3:4]))
