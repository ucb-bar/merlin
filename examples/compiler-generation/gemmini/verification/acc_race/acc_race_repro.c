/* Accumulator load-load ordering reproducer.
 *
 * Two loads that land in the SAME accumulator rows -- the first overwriting, the second accumulating --
 * are ordered by ISSUE in the load queue, not by COMPLETION. If the accumulating load's DMA data
 * returns first, the overwriting load lands on top of it and erases it, and the readout is the first
 * operand alone. Every test below reads the device result back and classifies each element as
 *   ok        the sum of both operands
 *   lhs_only  the first (overwriting) operand alone   <- the race's signature
 *   rhs_only  the second operand alone
 *   other     none of these
 * so a run says not only THAT an element is wrong but WHICH ordering produced it.
 *
 * Tests (each repeated REPS times on the same data, so a timing-dependent failure gets several chances):
 *   RACE cand07    mvin(lhs -> acc, overwrite); mvin2(rhs -> acc, accumulate); mvout   -- no ordering
 *   RACE fenced    the same with a fence between the two loads                         -- ordered
 *   RACE sameunit  both loads on load unit 0                                            -- no ordering
 *   VEND exact     the vendor tiled_resadd_auto at the model's view (ROWS x COLS28), scales 1
 *   VEND g19       the same call with the captured g19 scales and relu
 *   VEND perloop   the vendor call split so each accumulator loop is followed by a fence
 *   VEND view512   the vendor call on the 512-column view of the same bytes
 *
 * A per-16-column histogram of the failing elements in the 512-column view is printed for each test,
 * which is how "the final 32-wide column tile" is either reproduced or not.
 *
 * Build knobs: -DRACE_ROWS -DRACE_COLS -DVEND_ROWS -DREPS -DSKIP_VENDOR (the small build is what the
 * RTL simulators can afford; the default is what the FPGA runs).
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini_testutils.h"

#ifndef RACE_ROWS
#define RACE_ROWS 256
#endif
#ifndef RACE_COLS
#define RACE_COLS 512
#endif
#ifndef VEND_ELEMS
#define VEND_ELEMS 401408 /* the g19 tensor: 784 x 512 = 14336 x 28 */
#endif
#ifndef VEND_COLS_NARROW
#define VEND_COLS_NARROW 28
#endif
#ifndef REPS
#define REPS 4
#endif

#define ACC_BIT ((uint32_t)1 << (ADDR_LEN - 1))
#define ACCUM_BIT ((uint32_t)1 << (ADDR_LEN - 2))
#define WIDE 512 /* the 512-column view the histogram is reported in */
#define NBIN (WIDE / DIM)

static elem_t RA[RACE_ROWS * RACE_COLS] row_align(1);
static elem_t RB[RACE_ROWS * RACE_COLS] row_align(1);
static elem_t RC[RACE_ROWS * RACE_COLS] row_align(1);

#ifndef SKIP_VENDOR
static elem_t VA[VEND_ELEMS] row_align(1);
static elem_t VB[VEND_ELEMS] row_align(1);
static elem_t VC[VEND_ELEMS] row_align(1);
#endif

static uint32_t lcg = 12345u;
static int rnd(int lo, int hi) { lcg = lcg * 1664525u + 1013904223u; return lo + (int)((lcg >> 8) % (uint32_t)(hi - lo + 1)); }

static long long sat8(long long v) { return v > 127 ? 127 : (v < -128 ? -128 : v); }
static long long rne(float x) {
    long long f = (long long)x; if (x < 0 && (float)f != x) f -= 1;
    float rem = x - (float)f;
    if (rem > 0.5f) return f + 1; if (rem < 0.5f) return f;
    return (f % 2 == 0) ? f : f + 1;
}
static long long fin(long long v, int relu) { v = sat8(v); return (relu && v < 0) ? 0 : v; }
static long long absll(long long v) { return v < 0 ? -v : v; }

typedef struct { long long ok, lhs, rhs, oth; int bins[NBIN]; long long first; } tally_t;

/* Classify n elements. `tol` is the slack a scaled path is allowed; exact paths pass 0. */
static void classify(const elem_t *a, const elem_t *b, const elem_t *c, size_t n, float sa, float sb, int relu,
                     int tol, tally_t *t) {
    memset(t, 0, sizeof(*t)); t->first = -1;
    for (size_t i = 0; i < n; i++) {
        long long pa = rne((float)a[i] * sa), pb = rne((float)b[i] * sb), dev = c[i];
        long long full = fin(pa + pb, relu), lo = fin(pa, relu), ro = fin(pb, relu);
        if (absll(dev - full) <= tol) { t->ok++; continue; }
        if (absll(dev - lo) <= tol) t->lhs++;
        else if (absll(dev - ro) <= tol) t->rhs++;
        else t->oth++;
        t->bins[(i % WIDE) / DIM]++;
        if (t->first < 0) t->first = (long long)i;
    }
}

static void report(const char *kind, const char *name, int rep, const tally_t *t) {
    printf("%s %s rep=%d bad=%lld lhs_only=%lld rhs_only=%lld other=%lld first=%lld bins512=",
           kind, name, rep, t->lhs + t->rhs + t->oth, t->lhs, t->rhs, t->oth, t->first);
    for (int k = 0; k < NBIN; k++) printf(k ? ",%d" : "%d", t->bins[k]);
    printf("\n");
}

/* The cand_07 accumulate-add loop, verbatim in structure: one tile per DIMxDIM block, a rotating pool of
 * accumulator slots, the first operand with the accumulate bit CLEAR and the second with it SET. */
static void race_loop(int mode) {
    const size_t pool = ACC_ROWS / DIM;
    gemmini_extended4_config_ld(RACE_COLS * sizeof(elem_t), MVIN_SCALE_IDENTITY, true, DIM, 0);
    gemmini_extended4_config_ld(RACE_COLS * sizeof(elem_t), MVIN_SCALE_IDENTITY, true, DIM, 1);
    gemmini_config_ex(WS, 0, 0);
    gemmini_extended_config_st(RACE_COLS * sizeof(elem_t), NO_ACTIVATION, 1.0f);
    size_t idx = 0;
    for (size_t i = 0; i < RACE_ROWS / DIM; i++) {
        for (size_t j = 0; j < RACE_COLS / DIM; j++) {
            const uint32_t slot = (uint32_t)((idx % pool) * DIM); idx++;
            const size_t off = i * DIM * RACE_COLS + j * DIM;
            gemmini_extended_mvin(RA + off, ACC_BIT | slot, DIM, DIM);
            if (mode == 1) { gemmini_fence(); }
            if (mode == 2) { gemmini_extended_mvin(RB + off, ACC_BIT | ACCUM_BIT | slot, DIM, DIM); }
            else { gemmini_extended_mvin2(RB + off, ACC_BIT | ACCUM_BIT | slot, DIM, DIM); }
            gemmini_extended_mvout(RC + off, ACC_BIT | ACCUM_BIT | slot, DIM, DIM);
        }
    }
    gemmini_fence();
}

int main(void) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }
#endif
    gemmini_flush(0);
    printf("ACC_RACE begin race=%dx%d reps=%d\n", RACE_ROWS, RACE_COLS, REPS);

    for (size_t i = 0; i < RACE_ROWS * RACE_COLS; i++) { RA[i] = (elem_t)rnd(1, 40); RB[i] = (elem_t)rnd(41, 80); }
    static const char *const names[3] = {"cand07", "fenced", "sameunit"};
    tally_t t;
    for (int mode = 0; mode < 3; mode++) {
        for (int rep = 0; rep < REPS; rep++) {
            memset(RC, 0x55, sizeof(RC));
            race_loop(mode);
            classify(RA, RB, RC, RACE_ROWS * RACE_COLS, 1.0f, 1.0f, 0, 0, &t);
            report("RACE", names[mode], rep, &t);
        }
    }

#ifndef SKIP_VENDOR
    const size_t rows28 = VEND_ELEMS / VEND_COLS_NARROW;
    for (size_t i = 0; i < VEND_ELEMS; i++) { VA[i] = (elem_t)rnd(1, 40); VB[i] = (elem_t)rnd(41, 80); }
    for (int rep = 0; rep < REPS; rep++) {
        memset(VC, 0x55, sizeof(VC));
        tiled_resadd_auto(rows28, VEND_COLS_NARROW, 1.0f, 1.0f, 1.0f, VA, VB, VC, false, WS);
        classify(VA, VB, VC, VEND_ELEMS, 1.0f, 1.0f, 0, 0, &t);
        report("VEND", "exact", rep, &t);
    }
    for (int rep = 0; rep < REPS; rep++) {
        memset(VC, 0x55, sizeof(VC));
        const size_t blk = 224; /* the tile tiled_resadd_auto picks for this shape: one accumulator loop */
        for (size_t r = 0; r < rows28; r += blk)
            tiled_resadd_auto(r + blk <= rows28 ? blk : rows28 - r, VEND_COLS_NARROW, 1.0f, 1.0f, 1.0f, VA + r * VEND_COLS_NARROW,
                              VB + r * VEND_COLS_NARROW, VC + r * VEND_COLS_NARROW, false, WS);
        classify(VA, VB, VC, VEND_ELEMS, 1.0f, 1.0f, 0, 0, &t);
        report("VEND", "perloop", rep, &t);
    }
    for (int rep = 0; rep < REPS; rep++) {
        memset(VC, 0x55, sizeof(VC));
        tiled_resadd_auto(VEND_ELEMS / WIDE, WIDE, 1.0f, 1.0f, 1.0f, VA, VB, VC, false, WS);
        classify(VA, VB, VC, VEND_ELEMS, 1.0f, 1.0f, 0, 0, &t);
        report("VEND", "view512", rep, &t);
    }
    /* g19 as captured: signed stimulus, the two load scales, relu, readout scale 1. Classified with one
     * step of slack, since the load path's rounding is not what this probe is about. */
    for (size_t i = 0; i < VEND_ELEMS; i++) { VA[i] = (elem_t)rnd(-64, 63); VB[i] = (elem_t)rnd(-64, 63); }
    for (int rep = 0; rep < REPS; rep++) {
        memset(VC, 0x55, sizeof(VC));
        tiled_resadd_auto(rows28, VEND_COLS_NARROW, 0.8157085453198704f, 0.9234207574525486f, 1.0f, VA, VB, VC,
                          true, WS);
        classify(VA, VB, VC, VEND_ELEMS, 0.8157085453198704f, 0.9234207574525486f, 1, 1, &t);
        report("VEND", "g19", rep, &t);
    }
#endif
    printf("ACC_RACE end\n");
    exit(0);
}
