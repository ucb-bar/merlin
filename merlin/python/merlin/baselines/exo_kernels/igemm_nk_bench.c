/* On-board correctness + timing bench for the transpose-free int8 GEMMs vs the transposed EXO GEMM.
 *
 * Computes Y = A_i16[M,K] @ W_i8[N,K]^T three ways on the same random data:
 *   (0) TRANSPOSED baseline: glue widens+transposes W i8[N,K] -> Wt i16[K,N] (timed), then the EXO
 *       vwmacc GEMM igemm_nt_ref(Wt) — this is the cost the current arm pays.
 *   (A) igemm_nk_strided(W16[N,K])  — transpose-free strided-vwmacc (needs a contiguous i8->i16
 *       widen of W, timed as W_PREP; NO transpose scatter).
 *   (B) igemm_nk_dot(W16[N,K])      — transpose-free k-reduction dot.
 * Verifies A and B match the transposed baseline exactly (integer), and prints per-strategy ticks
 * (GEMM only) + the prep ticks (transpose vs contiguous-widen) so the runner picks the fastest and
 * confirms the transpose region is eliminated.
 *
 * Shapes via -DBM -DBN -DBK (default the lm_head-ish 8 x 4096 x 2048).
 */
#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "exo_igemm.h"   /* igemm_nt_ref(void*,M,N,K,int32_t*,const uint16_t*,const uint16_t*) — transposed */

void igemm_nk_strided(void*, int, int, int, int32_t*, const uint16_t*, const uint16_t*);
void igemm_nk_dot(void*, int, int, int, int32_t*, const uint16_t*, const uint16_t*);

#ifndef BM
#define BM 8
#endif
#ifndef BN
#define BN 4096
#endif
#ifndef BK
#define BK 2048
#endif

static inline uint64_t rd_time(void){ uint64_t t; __asm__ volatile("rdtime %0":"=r"(t)); return t; }

static int16_t X16[BM*BK];
static int8_t  W8[BN*BK];
static int16_t W16[BN*BK];      /* native [N,K] widened */
static int16_t WT16[BK*BN];     /* transposed [K,N] */
static int32_t Y0[BM*BN], YA[BM*BN], YB[BM*BN];

int main(void){
  uint32_t s=12345u;
  for(long i=0;i<(long)BM*BK;i++){ s=s*1103515245u+12345u; X16[i]=(int16_t)((int)((s>>16)&0xFF)-128); }
  for(long i=0;i<(long)BN*BK;i++){ s=s*1103515245u+12345u; W8[i]=(int8_t)((int)((s>>16)&0xFF)-128); }

  /* (0) transposed baseline: widen+transpose W i8[N,K] -> WT i16[K,N] (the scatter we pay today) */
  uint64_t t0=rd_time();
  for(int n=0;n<BN;n++){ const int8_t*wr=W8+(size_t)n*BK; for(int k=0;k<BK;k++) WT16[(size_t)k*BN+n]=(int16_t)wr[k]; }
  uint64_t t1=rd_time(); uint64_t prep_transpose=t1-t0;
  igemm_nt_ref(0, BM, BN, BK, Y0, (const uint16_t*)X16, (const uint16_t*)WT16);
  uint64_t t2=rd_time(); uint64_t gemm_transposed=t2-t1;

  /* transpose-free prep: contiguous i8->i16 widen of W (same [N,K] layout, streaming — no scatter) */
  uint64_t t3=rd_time();
  for(long i=0;i<(long)BN*BK;i++) W16[i]=(int16_t)W8[i];
  uint64_t t4=rd_time(); uint64_t prep_widen=t4-t3;

  /* (A) strided-vwmacc, native [N,K] */
  uint64_t t5=rd_time();
  igemm_nk_strided(0, BM, BN, BK, YA, (const uint16_t*)X16, (const uint16_t*)W16);
  uint64_t t6=rd_time(); uint64_t gemm_strided=t6-t5;

  /* (B) k-reduction dot, native [N,K] */
  uint64_t t7=rd_time();
  igemm_nk_dot(0, BM, BN, BK, YB, (const uint16_t*)X16, (const uint16_t*)W16);
  uint64_t t8=rd_time(); uint64_t gemm_dot=t8-t7;

  int errA=0, errB=0;
  for(long i=0;i<(long)BM*BN;i++){ if(YA[i]!=Y0[i]) errA++; if(YB[i]!=Y0[i]) errB++; }

  printf("NKBENCH shape %dx%dx%d prep_transpose %llu prep_widen %llu "
         "gemm_transposed %llu gemm_strided %llu gemm_dot %llu errA %d errB %d\n",
         BM, BN, BK, (unsigned long long)prep_transpose, (unsigned long long)prep_widen,
         (unsigned long long)gemm_transposed, (unsigned long long)gemm_strided,
         (unsigned long long)gemm_dot, errA, errB);
  return 0;
}
