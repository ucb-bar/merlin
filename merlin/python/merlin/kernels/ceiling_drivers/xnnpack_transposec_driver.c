// CEILING driver: XNNPACK RVV x32 TRANSPOSE microkernel (transposec), STANDALONE on K1.
// Transpose is the single largest BYTE-traffic op family across the model census
// (attention/norm reshapes), so it is raced even though it does zero arithmetic.
//
//   -DXNN_KERNEL_SRC=\"x32-transposec/gen/x32-transposec-8xv4-rvv.c\"
//   -DXNN_KERNEL_FN=xnn_x32_transposec_ukernel__8xv4_rvv
//   -DTR_R=256 -DTR_C=256   (input is (R,C) row-major -> output (C,R))
//
// The kernel transposes a block_height x block_width tile: block_height=R rows of
// block_width=C columns, input_stride = C*4 bytes, output (C rows x R cols),
// output_stride = R*4 bytes. inner_compute: only the ukernel call is timed.

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
#include "util.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/transpose.h"

#include XNN_KERNEL_SRC

#ifndef TR_R
#define TR_R 256
#endif
#ifndef TR_C
#define TR_C 256
#endif

static uint32_t IN[TR_R * TR_C];
static uint32_t OUT[TR_C * TR_R];

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  for (long i = 0; i < (long)TR_R * TR_C; i++) IN[i] = (uint32_t)(i * 2654435761u);
  for (long i = 0; i < (long)TR_C * TR_R; i++) OUT[i] = 0u;

  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  XNN_KERNEL_FN(IN, OUT,
                (size_t)TR_C * sizeof(uint32_t),   // input_stride  (bytes/input row)
                (size_t)TR_R * sizeof(uint32_t),   // output_stride (bytes/output row)
                (size_t)TR_C,                      // block_width
                (size_t)TR_R);                     // block_height
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);

  unsigned long cycles = c1 - c0;
  unsigned long instrs = i1 - i0;

  // verify: OUT[c,r] == IN[r,c]  (bit-exact, integer move)
  int errors = 0;
  for (long r = 0; r < TR_R && errors < 8; r++)
    for (long c = 0; c < TR_C; c++)
      if (OUT[c * TR_R + r] != IN[r * TR_C + c]) { errors++; break; }

  unsigned long long checksum = 0;
  for (long i = 0; i < (long)TR_C * TR_R; i++) checksum += OUT[i];

  printf("XNNPACK transposec R=%d C=%d\n", TR_R, TR_C);
  printf("CHECKSUM %lu (raw)\n", (unsigned long)(checksum & 0xffffffffUL));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
