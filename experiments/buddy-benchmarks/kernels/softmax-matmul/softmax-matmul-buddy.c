#include <stdint.h>
#include <stdio.h>

#include "include/gemmini.h"
#include "include/gemmini_testutils.h"

#define MAT_DIM_I 31
#define MAT_DIM_K 30
#define MAT_DIM_J 66

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

extern void _mlir_ciface_softmax_matmul(MemRef2D_i8 *a, MemRef2D_i8 *b,
                                        MemRef2D_i8 *c, MemRef2D_i32 *d);

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
  static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
  static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
  static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
  static acc_t full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_K; ++j) {
      full_A[i][j] = (rand() % 7) - 3;
    }
  }

  for (size_t i = 0; i < MAT_DIM_K; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j) {
      full_B[i][j] = (rand() % 7) - 3;
    }
  }

  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j) {
      full_D[i][j] = 0;
    }
  }

  long long a_checksum = 0;
  elem_t *a_ptr = &full_A[0][0];
  int a_elems = MAT_DIM_I * MAT_DIM_K;
  for (int i = 0; i < a_elems; ++i) {
    a_checksum += a_ptr[i];
  }
  long long b_checksum = 0;
  elem_t *b_ptr = &full_B[0][0];
  int b_elems = MAT_DIM_K * MAT_DIM_J;
  for (int i = 0; i < b_elems; ++i) {
    b_checksum += b_ptr[i];
  }
  long long d_checksum = 0;
  acc_t *d_ptr = &full_D[0][0];
  int d_elems = MAT_DIM_I * MAT_DIM_J;
  for (int i = 0; i < d_elems; ++i) {
    d_checksum += d_ptr[i];
  }
  printf("A checksum: %lld\n", a_checksum);
  printf("B checksum: %lld\n", b_checksum);
  printf("D checksum: %lld\n", d_checksum);

  MemRef2D_i8 a_ref = make_memref_i8(&full_A[0][0], MAT_DIM_I, MAT_DIM_K);
  MemRef2D_i8 b_ref = make_memref_i8(&full_B[0][0], MAT_DIM_K, MAT_DIM_J);
  MemRef2D_i8 c_ref = make_memref_i8(&full_C[0][0], MAT_DIM_I, MAT_DIM_J);
  MemRef2D_i32 d_ref = make_memref_i32(&full_D[0][0], MAT_DIM_I, MAT_DIM_J);

  gemmini_flush(0);
  uint64_t start = read_cycles();
  _mlir_ciface_softmax_matmul(&a_ref, &b_ref, &c_ref, &d_ref);
  gemmini_fence();
  uint64_t end = read_cycles();

  printf("Buddy softmax matmul cycles: %llu\n",
         (unsigned long long)(end - start));
  long long c_checksum = 0;
  elem_t *c_ptr = &full_C[0][0];
  int c_elems = MAT_DIM_I * MAT_DIM_J;
  for (int i = 0; i < c_elems; ++i) {
    c_checksum += c_ptr[i];
  }
  printf("Buddy output checksum: %lld\n", c_checksum);
  return 0;
}
