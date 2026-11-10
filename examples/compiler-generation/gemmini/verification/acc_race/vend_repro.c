/* The vendor residual add, alone, on operands that arrive COLD from DRAM.
 *
 * acc_race_repro.c generates its stimulus on the core, so the operands sit dirty in the cache hierarchy
 * when the accelerator reads them and never reach the DRAM model -- a reordering memory model can then
 * perturb nothing. Here the operands are initialised data in the ELF (vend_data.h, written by
 * gen_vend_data.py), loaded straight into DRAM, read exactly once. That is the situation of a capsule's
 * inputs and of a whole-model layer's activations.
 *
 * One call of tiled_resadd_auto(VEND_ROWS, 512) at unit scales: the tiler picks 49x96 tiles, so every
 * row ends in a 32-wide column tile -- the tile the FireSim failures sit in. Each element is classified
 * with integer arithmetic only (ok / lhs_only / rhs_only / other), and the columns of the failures are
 * histogrammed in 16-wide bins.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "vend_data.h" /* VEND_ROWS, VEND_COLS, VA[], VB[] */

static elem_t VC[VEND_ROWS * VEND_COLS] row_align(1);

int main(void) {
    gemmini_flush(0);
    tiled_resadd_auto(VEND_ROWS, VEND_COLS, 1.0f, 1.0f, 1.0f, VA, VB, VC, false, WS);
    long long ok = 0, lhs = 0, rhs = 0, oth = 0, first = -1;
    int bins[VEND_COLS / DIM] = {0};
    for (size_t i = 0; i < (size_t)VEND_ROWS * VEND_COLS; i++) {
        int a = VA[i], b = VB[i], s = a + b, d = VC[i];
        if (s > 127) s = 127;
        if (d == s) { ok++; continue; }
        if (d == a) lhs++; else if (d == b) rhs++; else oth++;
        bins[(i % VEND_COLS) / DIM]++;
        if (first < 0) first = (long long)i;
    }
    printf("VEND cold rows=%d bad=%lld lhs_only=%lld rhs_only=%lld other=%lld first=%lld bins=", VEND_ROWS,
           lhs + rhs + oth, lhs, rhs, oth, first);
    for (int k = 0; k < VEND_COLS / DIM; k++) printf(k ? ",%d" : "%d", bins[k]);
    printf("\n");
    exit(0);
}
