"""Copy a bare-metal Gemmini harness and make its residual add ORDER its two accumulator loads.

The vendor `sp_tiled_resadd` hands a tile to the LOOP_WS unroller in residual-add mode, which issues
every overwriting load of operand A and then every accumulating load of operand B into the same
accumulator rows -- ordered by issue, not by completion. This rewrite replaces that single unroller call
with the same three phases issued explicitly (A on load unit 0, B on load unit 1, readout), with a fence
between A and B. Scales, activation and readout scale are unchanged (they are configured by the caller,
`tiled_resadd`, exactly as before), so the only difference from the vendor library is the ordering.

Usage: fence_vendor_resadd.py <harness_in> <harness_out>

The copy is a new tree; the input harness is never edited.
"""

import shutil
import sys
from pathlib import Path

FENCED = """
static void sp_tiled_resadd_fenced(const size_t I, const size_t J,
        const elem_t * A, const elem_t * B, elem_t * C,
        size_t A_row_stride, size_t B_row_stride, size_t C_row_stride) {
    const uint32_t acc = (uint32_t)1 << (ADDR_LEN - 1);
    const uint32_t accum = (uint32_t)1 << (ADDR_LEN - 2);
    const size_t nb = J / DIM + (J % DIM != 0);
    for (size_t i = 0; i < I; i += DIM)
        for (size_t j = 0; j < J; j += DIM) {
            const size_t rows = i + DIM <= I ? DIM : I - i, cols = j + DIM <= J ? DIM : J - j;
            gemmini_extended_mvin(A + i * A_row_stride + j, acc | (uint32_t)((i / DIM) * nb * DIM + j), cols, rows);
        }
    gemmini_fence();
    for (size_t i = 0; i < I; i += DIM)
        for (size_t j = 0; j < J; j += DIM) {
            const size_t rows = i + DIM <= I ? DIM : I - i, cols = j + DIM <= J ? DIM : J - j;
            gemmini_extended_mvin2(B + i * B_row_stride + j, acc | accum | (uint32_t)((i / DIM) * nb * DIM + j), cols, rows);
        }
    for (size_t i = 0; i < I; i += DIM)
        for (size_t j = 0; j < J; j += DIM) {
            const size_t rows = i + DIM <= I ? DIM : I - i, cols = j + DIM <= J ? DIM : J - j;
            gemmini_extended_mvout(C + i * C_row_stride + j, acc | (uint32_t)((i / DIM) * nb * DIM + j), cols, rows);
        }
}

"""

CALL = "sp_tiled_resadd_fenced(I, J, A, B, C, A_row_stride, B_row_stride, C_row_stride);"


def fence_resadd(text: str) -> str:
    """The header text with the residual add's single unroller call replaced by the fenced phases.

    Refuses a header that does not define the vendor residual add exactly once, or whose definition
    does not hand the tile to the loop unroller: patching anything else would build a binary that
    differs from the vendor library in more than the ordering.
    """
    head = "static void sp_tiled_resadd(const size_t I"
    if text.count(head) != 1:
        raise ValueError(f"expected one sp_tiled_resadd definition, found {text.count(head)}")
    start = text.index(head)
    body_end = text.find("\n}\n", start)
    call_at = text.find("gemmini_loop_ws(", start)
    if call_at < 0 or (body_end >= 0 and call_at > body_end):
        raise ValueError("sp_tiled_resadd does not call gemmini_loop_ws")
    call_end = text.index(";", call_at) + 1
    return text[:start] + FENCED + text[start:call_at] + CALL + text[call_end:]


def main(src: str, dst: str) -> None:
    shutil.copytree(src, dst)
    hdr = Path(dst) / "include" / "gemmini.h"
    hdr.write_text(fence_resadd(hdr.read_text()))
    print(f"patched {hdr}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
