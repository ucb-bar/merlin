#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini.h"
#include "parameters1.h"

typedef struct {
  elem_t *basePtr;
  elem_t *data;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2D_i8;

typedef struct {
  acc_t *basePtr;
  acc_t *data;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2D_i32;

extern void _mlir_ciface_mlp1(MemRef2D_i8 *a0, MemRef2D_i8 *w0,
                              MemRef2D_i8 *c0, MemRef2D_i32 *d0,
                              MemRef2D_i8 *w1, MemRef2D_i8 *c1,
                              MemRef2D_i32 *d1, MemRef2D_i8 *w2,
                              MemRef2D_i8 *c2, MemRef2D_i32 *d2,
                              MemRef2D_i8 *w3, MemRef2D_i8 *c3,
                              MemRef2D_i32 *d3, MemRef2D_i8 *w4,
                              MemRef2D_i8 *c4, MemRef2D_i32 *d4,
                              MemRef2D_i8 *w5, MemRef2D_i8 *c5,
                              MemRef2D_i32 *d5);

static uint32_t lcg_state = 777;
static inline elem_t next_elem(void) {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return (elem_t)((lcg_state >> 24) % 5) - 2;
}

static void init_random_i8(elem_t *buf, int len) {
  for (int i = 0; i < len; ++i) {
    buf[i] = next_elem();
  }
}

static inline uint64_t read_cycles(void) {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

static MemRef2D_i8 make_memref_i8(elem_t *base, int64_t rows, int64_t cols) {
  MemRef2D_i8 ref;
  ref.basePtr = base;
  ref.data = base;
  ref.offset = 0;
  ref.sizes[0] = rows;
  ref.sizes[1] = cols;
  ref.strides[1] = 1;
  ref.strides[0] = cols;
  return ref;
}

static MemRef2D_i32 make_memref_i32(acc_t *base, int64_t rows, int64_t cols) {
  MemRef2D_i32 ref;
  ref.basePtr = base;
  ref.data = base;
  ref.offset = 0;
  ref.sizes[0] = rows;
  ref.sizes[1] = cols;
  ref.strides[1] = 1;
  ref.strides[0] = cols;
  return ref;
}

static acc_t d0_bias[64][2560] row_align_acc(1) = {0};
static acc_t d1_bias[64][2048] row_align_acc(1) = {0};
static acc_t d2_bias[64][1536] row_align_acc(1) = {0};
static acc_t d3_bias[64][1024] row_align_acc(1) = {0};
static acc_t d4_bias[64][512] row_align_acc(1) = {0};
static acc_t d5_bias[64][64] row_align_acc(1) = {0};

int main(void) {
  lcg_state = 777;
  init_random_i8(&input_mat[0][0], (int)(sizeof(input_mat) / sizeof(elem_t)));
  init_random_i8(&weights0[0][0], (int)(sizeof(weights0) / sizeof(elem_t)));
  init_random_i8(&weights1[0][0], (int)(sizeof(weights1) / sizeof(elem_t)));
  init_random_i8(&weights2[0][0], (int)(sizeof(weights2) / sizeof(elem_t)));
  init_random_i8(&weights3[0][0], (int)(sizeof(weights3) / sizeof(elem_t)));
  init_random_i8(&weights4[0][0], (int)(sizeof(weights4) / sizeof(elem_t)));
  init_random_i8(&weights5[0][0], (int)(sizeof(weights5) / sizeof(elem_t)));

  memset(inter_results0, 0, sizeof(inter_results0));
  memset(inter_results1, 0, sizeof(inter_results1));
  memset(inter_results2, 0, sizeof(inter_results2));
  memset(inter_results3, 0, sizeof(inter_results3));
  memset(inter_results4, 0, sizeof(inter_results4));
  memset(inter_results5, 0, sizeof(inter_results5));
  memset(d0_bias, 0, sizeof(d0_bias));
  memset(d1_bias, 0, sizeof(d1_bias));
  memset(d2_bias, 0, sizeof(d2_bias));
  memset(d3_bias, 0, sizeof(d3_bias));
  memset(d4_bias, 0, sizeof(d4_bias));
  memset(d5_bias, 0, sizeof(d5_bias));

  MemRef2D_i8 a0_ref = make_memref_i8(&input_mat[0][0], 64, 832);
  MemRef2D_i8 w0_ref = make_memref_i8(&weights0[0][0], 832, 2560);
  MemRef2D_i8 c0_ref = make_memref_i8(&inter_results0[0][0], 64, 2560);
  MemRef2D_i32 d0_ref = make_memref_i32(&d0_bias[0][0], 64, 2560);

  MemRef2D_i8 w1_ref = make_memref_i8(&weights1[0][0], 2560, 2048);
  MemRef2D_i8 c1_ref = make_memref_i8(&inter_results1[0][0], 64, 2048);
  MemRef2D_i32 d1_ref = make_memref_i32(&d1_bias[0][0], 64, 2048);

  MemRef2D_i8 w2_ref = make_memref_i8(&weights2[0][0], 2048, 1536);
  MemRef2D_i8 c2_ref = make_memref_i8(&inter_results2[0][0], 64, 1536);
  MemRef2D_i32 d2_ref = make_memref_i32(&d2_bias[0][0], 64, 1536);

  MemRef2D_i8 w3_ref = make_memref_i8(&weights3[0][0], 1536, 1024);
  MemRef2D_i8 c3_ref = make_memref_i8(&inter_results3[0][0], 64, 1024);
  MemRef2D_i32 d3_ref = make_memref_i32(&d3_bias[0][0], 64, 1024);

  MemRef2D_i8 w4_ref = make_memref_i8(&weights4[0][0], 1024, 512);
  MemRef2D_i8 c4_ref = make_memref_i8(&inter_results4[0][0], 64, 512);
  MemRef2D_i32 d4_ref = make_memref_i32(&d4_bias[0][0], 64, 512);

  MemRef2D_i8 w5_ref = make_memref_i8(&weights5[0][0], 512, 64);
  MemRef2D_i8 c5_ref = make_memref_i8(&inter_results5[0][0], 64, 64);
  MemRef2D_i32 d5_ref = make_memref_i32(&d5_bias[0][0], 64, 64);

  gemmini_flush(0);

  uint64_t start = read_cycles();
  _mlir_ciface_mlp1(&a0_ref, &w0_ref, &c0_ref, &d0_ref,
                    &w1_ref, &c1_ref, &d1_ref,
                    &w2_ref, &c2_ref, &d2_ref,
                    &w3_ref, &c3_ref, &d3_ref,
                    &w4_ref, &c4_ref, &d4_ref,
                    &w5_ref, &c5_ref, &d5_ref);
  gemmini_fence();
  uint64_t end = read_cycles();

  printf("Buddy mlp1 cycles: %llu\n", (unsigned long long)(end - start));
  long long checksum = 0;
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      checksum += inter_results5[i][j];
    }
  }
  printf("Buddy mlp1 output checksum: %lld\n", checksum);
  return 0;
}
