"""Exact widths for the bounded scalar RV64 store forms in commit evidence.

This decodes instruction effects only. It supplies no address, committed value,
source ownership, initialization or execution authority. Vector stores and
unrecognized/reserved encodings require a separate effect decoder.
"""


def rv64_scalar_store_width(word: int) -> int:
    """Decode integer/FP stores, writing atomics and RV64 compressed stores.

    The caller must independently require a complete committed memory value of
    this width. In particular, a load-reserved instruction is never a store.
    """
    if type(word) is not int or not 0 <= word < 1 << 32:
        raise ValueError("RV64 scalar store needs a bounded original instruction")
    if word & 3 == 3:
        opcode, funct3 = word & 127, word >> 12 & 7
        if opcode == 0x23 and funct3 in (0, 1, 2, 3):
            return 1 << funct3
        if opcode == 0x27 and funct3 in (2, 3):
            return 1 << funct3
        if (
            opcode == 0x2F
            and funct3 in (2, 3)
            and word >> 27 in (0, 1, 3, 4, 8, 12, 16, 20, 24, 28)
        ):
            return 1 << funct3
    elif word < 1 << 16 and word & 3 in (0, 2):
        # RV64 C.FSD/C.FSDSP, C.SW/C.SWSP, C.SD/C.SDSP.
        funct3 = word >> 13 & 7
        if funct3 in (5, 6, 7):
            return 4 if funct3 == 6 else 8
    raise ValueError("unresolved RV64 scalar store instruction effect")
