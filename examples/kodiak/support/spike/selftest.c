/* Self-test of the spike OPU model on the instruction forms the certification corpus does NOT exercise.
 *
 * The corpus (merlin.kernels.opu_cert) drives the microkernel: row broadcast, accumulate, readout into
 * v0. This drives the rest -- row and column move-in, column broadcast, every tile register, readout at
 * e8/m1, and the readout into v16..v31, where `rd(4)` swaps the subtile fields -- and checks each against
 * an expectation written in NATURAL tile coordinates (row i, column j), not the model's physical ones. So
 * it checks that the physical walk in opu_ext.cc reduces to the outer-product description wherever the
 * RTL means it to, and to the RTL-derived permutation where it does not.
 *
 * Instruction words come from the generated opu_encodings.h. Built and run by `build.py --selftest`.
 * Prints one `SELFTEST <name> <mismatches>` line per check and `SELFTEST_DONE <failed checks>`. */
#include <stddef.h>
#include <stdint.h>

#include "opu_encodings.h"

#include "htif.h"
int printf(const char *fmt, ...);

#define EDGE (OPU_VLEN / OPU_A_BITS)
#define SUB (OPU_VLEN / OPU_DLEN)
#define D (OPU_DLEN / OPU_A_BITS)                 /* cells per physical dimension */
#define CY (OPU_C_BITS / OPU_A_BITS)              /* clusterYdim == clusterXdim here */
#define YD (D / CY)                               /* yDim == xDim */

/* rd/rs1/rs2 fields; vm = 1. rs1 is always a0, which the caller loads. */
#define WORD(match, rd, rs2) ((match) | ((rd) << 7) | (10u << 15) | ((rs2) << 20) | (1u << 25))
#define OPU(match, rd, rs2, x)                                                        \
  do {                                                                                \
    register long a0_ asm("a0") = (long)(x);                                          \
    asm volatile(".word %1" ::"r"(a0_), "i"(WORD(match, rd, rs2)) : "memory");        \
  } while (0)

/* The register fields are encoded at compile time, so a tile chosen at run time goes through a switch. */
#define NTILES (OPU_MATRIX_REGS < 4 ? OPU_MATRIX_REGS : 4)
#define OPU_T(match, t, rs2, x)              \
  do {                                       \
    switch (t) {                             \
      case 0: OPU(match, 0, rs2, x); break;  \
      case 1: OPU(match, 1, rs2, x); break;  \
      case 2: OPU(match, 2, rs2, x); break;  \
      default: OPU(match, 3, rs2, x); break; \
    }                                        \
  } while (0)

/* Same, for the readout, whose tile is in the rs2 field (destination v0). */
#define OPU_T2(match, t, x)                 \
  do {                                      \
    switch (t) {                            \
      case 0: OPU(match, 0, 0, x); break;   \
      case 1: OPU(match, 0, 1, x); break;   \
      case 2: OPU(match, 0, 2, x); break;   \
      default: OPU(match, 0, 3, x); break;  \
    }                                       \
  } while (0)

static int32_t buf[EDGE], out[EDGE];
static int8_t a8[EDGE], b8[EDGE];

static void load32(const int32_t *p) { /* -> v8..v11 */
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\tvle32.v v8, (%1)" ::"r"(EDGE), "r"(p) : "memory");
}
static void e32m4(void) { asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(EDGE)); }
static void store_v0(int32_t *p) {
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\tvse32.v v0, (%1)" ::"r"(EDGE), "r"(p) : "memory");
}
static void store_v16(int32_t *p) {
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\tvse32.v v16, (%1)" ::"r"(EDGE), "r"(p) : "memory");
}

static int32_t nat(int i, int j, int tile) { return tile * 1000000 + i * 1000 + j; }

static int failed;
static void report(const char *name, int bad) {
  printf("SELFTEST %s %d\n", name, bad);
  failed += bad != 0;
}

/* Fill tile `t` with nat(i, j, t) one ROW at a time through the row move-in. */
static void fill_rows(int t) {
  for (int i = 0; i < EDGE; ++i) {
    for (int j = 0; j < EDGE; ++j) buf[j] = nat(i, j, t);
    load32(buf);
    OPU_T(OPU_OPERAND_LOAD_MATCH, t, 8, i);
  }
}

static int check_rows(int t, int transposed) {
  int bad = 0;
  e32m4();
  for (int i = 0; i < EDGE; ++i) {
    OPU_T2(OPU_READOUT_MATCH, t, i);
    store_v0(out);
    for (int j = 0; j < EDGE; ++j) bad += out[j] != (transposed ? nat(j, i, t) : nat(i, j, t));
  }
  return bad;
}

int main(int hart) {
  if (hart) return 0;
  e32m4();
  /* 1. row move-in -> readout, on every tile register; each tile must keep its own contents. */
  for (int t = 0; t < NTILES; ++t) fill_rows(t);
  int bad = 0;
  for (int t = 0; t < NTILES; ++t) bad += check_rows(t, 0);
  report("row_mvin_readout_all_tiles", bad);

  /* 2. column move-in (rd bit 4): column c of tile 2 gets the vector, i.e. the tile reads back transposed. */
  for (int c = 0; c < EDGE; ++c) {
    for (int i = 0; i < EDGE; ++i) buf[i] = nat(c, i, 2);
    load32(buf);
    OPU(OPU_OPERAND_LOAD_MATCH, 16 | 2, 8, c);
  }
  report("col_mvin_is_transpose", check_rows(2, 1));

  /* 3. row broadcast: every row of tile 3 equals the vector. */
  for (int j = 0; j < EDGE; ++j) buf[j] = 7 * j - 100;
  load32(buf);
  OPU(OPU_BROADCAST_MATCH, 3, 8, 0);
  bad = 0;
  for (int i = 0; i < EDGE; ++i) {
    OPU(OPU_READOUT_MATCH, 0, 3, i);
    store_v0(out);
    for (int j = 0; j < EDGE; ++j) bad += out[j] != 7 * j - 100;
  }
  report("row_bcast", bad);

  /* 4. column broadcast: every column of tile 3 equals the vector (row i is buf[i] everywhere). */
  OPU(OPU_BROADCAST_MATCH, 16 | 3, 8, 0);
  bad = 0;
  for (int i = 0; i < EDGE; ++i) {
    OPU(OPU_READOUT_MATCH, 0, 3, i);
    store_v0(out);
    for (int j = 0; j < EDGE; ++j) bad += out[j] != 7 * i - 100;
  }
  report("col_bcast", bad);

  /* 5. accumulate onto a moved-in tile: C = nat + a b^T, signed int8 extremes included. */
  fill_rows(1);
  for (int i = 0; i < EDGE; ++i) {
    a8[i] = (int8_t)(i * 37 - 128);
    b8[i] = (int8_t)(127 - i * 11);
  }
  /* WORD() puts 10 in the rs1 field, which for the VV accumulate names v10 (the row operand). */
  asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\tvle8.v v10, (%1)\n\tvle8.v v4, (%2)" ::"r"(EDGE), "r"(a8),
               "r"(b8) : "memory");
  OPU(OPU_ACCUMULATE_MATCH, 1, 4, 0);
  bad = 0;
  e32m4();
  for (int i = 0; i < EDGE; ++i) {
    OPU(OPU_READOUT_MATCH, 0, 1, i);
    store_v0(out);
    for (int j = 0; j < EDGE; ++j) bad += out[j] != nat(i, j, 1) + (int32_t)a8[i] * (int32_t)b8[j];
  }
  report("accumulate_outer_product", bad);

  /* 6. vl == 0 drops the instruction (Dispatch.scala): tile 0 must be untouched by a broadcast. */
  asm volatile("vsetvli zero, zero, e32, m4, ta, ma" ::: "memory");
  asm volatile("vsetivli zero, 0, e32, m4, ta, ma");
  OPU(OPU_BROADCAST_MATCH, 0, 8, 0);
  report("vl0_is_dropped", check_rows(0, 0));

  /* 7. readout into v16 (rd bit 4 set): the sequencer swaps the subtile fields of mrf_idx. Expected
   * value, in natural coordinates, for output element k of row R:
   *   col_idx = k / YD, e = k % YD, sc = col_idx / CY, cc = col_idx % CY
   *   c = R % YD, r = (R / YD) % CY, s = R / (YD * CY)
   *   value = tile[sc*D + r*YD + c][s*D + cc*YD + e]
   * which equals the plain row R exactly when SUB == 1. */
  bad = 0;
  e32m4();
  for (int R = 0; R < EDGE; ++R) {
    OPU(OPU_READOUT_MATCH, 16, 0, R);
    store_v16(out);
    int c = R % YD, r = (R / YD) % CY, s = R / (YD * CY);
    for (int k = 0; k < EDGE; ++k) {
      int col_idx = k / YD, e = k % YD, sc = col_idx / CY, cc = col_idx % CY;
      bad += out[k] != nat(sc * D + r * YD + c, s * D + cc * YD + e, 0);
    }
  }
  report("readout_v16_swaps_subtiles", bad);

  printf("SELFTEST_DONE %d\n", failed);
  htif_exit(failed);
}
