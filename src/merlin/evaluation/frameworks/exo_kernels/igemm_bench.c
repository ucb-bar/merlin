/* Standalone int8 vwmacc GEMM micro-benchmark for on-board autotuning (K1, rv64gcv).
 *
 * Times igemm_nt_ref (from the exocc-emitted exo_igemm.c, linked per candidate KU) on a fixed
 * representative shape and prints the rdtime ticks. The autotuner (merlin.baselines.exo) compiles
 * one binary per KU candidate, runs each once under board_lock, and keeps the fastest — a bounded
 * empirical search (RVV-audit + measured cycles), not a cost model.
 *
 * Shape via -DBM= -DBN= -DBK= (default the lm_head-ish 8 x 4096 x 2048). Inputs are deterministic
 * pseudo-random i16 in the i8 range so timing is representative; correctness is checked separately
 * by the whole-model gate, so this only reports ticks + a checksum (guards dead-code elimination).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "exo_igemm.h"   /* void igemm_nt_ref(void*, M,N,K, int32_t* Y, const uint16_t* X, const uint16_t* Wt) */

#ifndef BM
#define BM 8
#endif
#ifndef BN
#define BN 4096
#endif
#ifndef BK
#define BK 2048
#endif
#ifndef REPS
#define REPS 4
#endif

static inline uint64_t rd_time(void){ uint64_t t; __asm__ volatile("rdtime %0":"=r"(t)); return t; }

int main(void){
  static int16_t X[BM*BK];
  static int16_t Wt[BK*BN];
  static int32_t Y[BM*BN];
  uint32_t s=12345u;
  for(long i=0;i<(long)BM*BK;i++){ s=s*1103515245u+12345u; X[i]=(int16_t)((int)((s>>16)&0xFF)-128); }
  for(long i=0;i<(long)BK*BN;i++){ s=s*1103515245u+12345u; Wt[i]=(int16_t)((int)((s>>16)&0xFF)-128); }
  /* warm + time REPS calls */
  igemm_nt_ref(0, BM, BN, BK, Y, (const uint16_t*)X, (const uint16_t*)Wt);
  uint64_t t0=rd_time();
  for(int r=0;r<REPS;r++) igemm_nt_ref(0, BM, BN, BK, Y, (const uint16_t*)X, (const uint16_t*)Wt);
  uint64_t t1=rd_time();
  long long cs=0; for(long i=0;i<(long)BM*BN;i++) cs+=Y[i];
  printf("BENCH_TICKS %llu reps %d shape %dx%dx%d checksum %lld\n",
         (unsigned long long)(t1-t0), REPS, BM, BN, BK, cs);
  return 0;
}
