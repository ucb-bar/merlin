#include <stdint.h>
#include <stdio.h>

#include "include/gemmini.h"
#include "parameters2.h"

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

extern void _mlir_ciface_mlp2(MemRef2D_i8 *a0, MemRef2D_i8 *w0,
                              MemRef2D_i8 *c0, MemRef2D_i32 *d0,
                              MemRef2D_i8 *w1, MemRef2D_i8 *c1,
                              MemRef2D_i32 *d1);

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

static acc_t d0_bias[64][832] row_align_acc(1) = {0};
static acc_t d1_bias[64][64] row_align_acc(1) = {0};

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

int main(void) {
  lcg_state = 777;
  init_random_i8(&input_mat[0][0], (int)(sizeof(input_mat) / sizeof(elem_t)));
  init_random_i8(&weights0[0][0], (int)(sizeof(weights0) / sizeof(elem_t)));
  init_random_i8(&weights1[0][0], (int)(sizeof(weights1) / sizeof(elem_t)));

  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 832; ++j) {
      inter_results0[i][j] = 0;
      d0_bias[i][j] = 0;
    }
  }
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      inter_results1[i][j] = 0;
      d1_bias[i][j] = 0;
    }
  }

  MemRef2D_i8 a0_ref = make_memref_i8(&input_mat[0][0], 64, 832);
  MemRef2D_i8 w0_ref = make_memref_i8(&weights0[0][0], 832, 832);
  MemRef2D_i8 c0_ref = make_memref_i8(&inter_results0[0][0], 64, 832);
  MemRef2D_i32 d0_ref = make_memref_i32(&d0_bias[0][0], 64, 832);
  MemRef2D_i8 w1_ref = make_memref_i8(&weights1[0][0], 832, 64);
  MemRef2D_i8 c1_ref = make_memref_i8(&inter_results1[0][0], 64, 64);
  MemRef2D_i32 d1_ref = make_memref_i32(&d1_bias[0][0], 64, 64);

  gemmini_flush(0);

  uint64_t start = read_cycles();
  _mlir_ciface_mlp2(&a0_ref, &w0_ref, &c0_ref, &d0_ref,
                    &w1_ref, &c1_ref, &d1_ref);
  gemmini_fence();
  uint64_t end = read_cycles();

  printf("Buddy mlp2 cycles: %llu\n", (unsigned long long)(end - start));
  long long checksum = 0;
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      checksum += inter_results1[i][j];
    }
  }
  printf("Buddy mlp2 output checksum: %lld\n", checksum);
  return 0;
}
