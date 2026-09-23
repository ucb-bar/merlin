// CEILING driver for OURS fp16 GEMM (Merlin RVV codegen, fp16_f32acc datapath), measured
// STANDALONE on the SpacemiT K1 with the SAME inner-compute / rdtime protocol as the f32
// ours_gemm_driver.c. Separate from that driver because fp16 differs in two load-bearing ways:
//
//   1. DTYPE. The compiled `_mlir_ciface_forward` takes f16 operands and writes an f16 output
//      (the model is authored `tensor<MxKxf16> x tensor<KxNxf16> -> tensor<MxNxf16>`, then the
//      lowering rewrites the reduction to accumulate in f32 and truncf back to f16). So A, B and
//      OUT are 16-bit; the embedded inputs (model_io.h) are the RAW f16 bit patterns. We read them
//      as uint16 and widen with a portable bit-twiddle (no `_Float16` language dependency, so the
//      driver compiles under plain -march=rv64gcv).
//
//   2. GATE. fp16 cannot be bit-exact, and the int8-tier AGGREGATE gate (cos + rel-L2) ACCEPTS a
//      genuinely broken f16-ACCUMULATING kernel (measured: cos 0.99999858, rel-L2 0.0017, yet off
//      by 1209% on individual elements). So this driver gates PER-ELEMENT against the f64-exact
//      reference computed from the f16 operands:  cos > 0.9999 AND rel-L2 < 1e-2 AND max-rel < 0.05.
//      A correctly f32-accumulated result rounded once to f16 carries <= ~1 ulp (max-rel ~9e-4);
//      an f16-accumulating kernel misses by ~240x. Fail-closed: gate not met -> VERIFY FAIL.

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "util.h"            // read_csr(mcycle)->rdtime, read_csr(minstret)->perf_event
#include "merlin_model.h"
#include "model_gen.h"       // MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_OUT_ELEMS, ...
#include "model_io.h"        // merlin_in_0 / merlin_in_1 (raw f16 bit patterns), MERLIN_INPUT_PTR

#ifndef GEMM_M
#error "shape must be injected via -DGEMM_M= -DGEMM_N= -DGEMM_K="
#endif
#define M GEMM_M
#define N GEMM_N
#define K GEMM_K

// IEEE-754 binary16 -> double, exact. Portable (no _Float16); handles sub/normals, Inf/NaN.
static double h2d(uint16_t h) {
  uint32_t sign = (uint32_t)(h >> 15) & 1u;
  uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
  uint32_t man  = (uint32_t)h & 0x3FFu;
  double v;
  if (exp == 0u) {
    v = ldexp((double)man / 1024.0, -14);          // subnormal (or zero)
  } else if (exp == 0x1Fu) {
    v = man ? (0.0 / 0.0) : (1.0 / 0.0);           // NaN / Inf
  } else {
    v = ldexp(1.0 + (double)man / 1024.0, (int)exp - 15);
  }
  return sign ? -v : v;
}

static uint16_t OUT[MERLIN_OUT_ELEMS];             // f16 storage the kernel writes
static double    Cref[MERLIN_OUT_ELEMS];           // f64-exact reference over the f16 operands
static merlin_descriptor_t DESCS[MERLIN_N_ARGS];

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  const uint16_t* A = (const uint16_t*)merlin_in_0;   // M x K, row-major, f16 bits
  const uint16_t* B = (const uint16_t*)merlin_in_1;   // K x N, row-major, f16 bits

  // ---- f64-exact reference C[m,n] = sum_k h2d(A)*h2d(B), BEFORE timing ------
  // f64 accumulation over the SAME f16 operand values the kernel sees. This is the reference the
  // per-element gate measures against (NOT an f16- or f32-accumulating recompute, so it can expose
  // a kernel that itself accumulates in f16).
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      double acc = 0.0;
      for (int k = 0; k < K; k++) acc += h2d(A[m * K + k]) * h2d(B[k * N + n]);
      Cref[m * N + n] = acc;
    }

  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) OUT[i] = 0;

  // ---- descriptors OUTSIDE timing (mirror the expert pre-pack; see f32 driver) --
  void* desc_ptrs[MERLIN_N_ARGS];
  for (int i = 0; i < MERLIN_N_ARGS; i++) {
    void* data = 0;
    switch (MERLIN_ARGS[i].kind) {
      case MERLIN_INPUT:  data = MERLIN_INPUT_PTR[i]; break;
      case MERLIN_OUTPUT: data = (void*)OUT; break;
      default: break;
    }
    merlin_descriptor_t* d = &DESCS[i];
    d->allocated = data; d->aligned = data; d->offset = 0;
    long stride = 1;
    for (int r = MERLIN_ARGS[i].rank - 1; r >= 0; r--) {
      d->sizes[r] = MERLIN_ARGS[i].dims[r];
      d->strides[r] = stride;
      stride *= MERLIN_ARGS[i].dims[r];
    }
    desc_ptrs[i] = d;
  }

  // ---- fill-only baseline (zero the M*N f16 output), timed for subtraction -----
  static volatile uint16_t fill_sink;
  unsigned long f0 = read_csr(mcycle);
  unsigned long fi0 = read_csr(minstret);
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) OUT[i] = 0;
  unsigned long fi1 = read_csr(minstret);
  unsigned long f1 = read_csr(mcycle);
  fill_sink = OUT[0];
  unsigned long fill_cycles = f1 - f0;
  unsigned long fill_instrs = fi1 - fi0;

  // ---- TIMED region: the compiled compute (f32-acc fill + matmul + truncf) -----
  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  merlin_invoke(desc_ptrs);
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);

  unsigned long cycles_full = c1 - c0;
  unsigned long instrs_full = i1 - i0;
  unsigned long cycles = (cycles_full > fill_cycles) ? (cycles_full - fill_cycles) : 0;
  unsigned long instrs = (instrs_full > fill_instrs) ? (instrs_full - fill_instrs) : 0;

  // ---- PER-ELEMENT gate vs the f64-exact reference ------------------------------
  double dot = 0.0, no = 0.0, nr = 0.0, num = 0.0, maxrel = 0.0;
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) {
    double o = h2d(OUT[i]);
    double r = Cref[i];
    dot += o * r; no += o * o; nr += r * r;
    double d = o - r; if (d < 0) d = -d;
    num += d * d;
    double denom = fabs(r); if (denom < 1e-3) denom = 1e-3;
    double rel = d / denom;
    if (rel > maxrel) maxrel = rel;
  }
  double cos = dot / (sqrt(no) * sqrt(nr) + 1e-12);
  double rel_l2 = sqrt(num) / (sqrt(nr) + 1e-12);
  int pass = (cos > 0.9999) && (rel_l2 < 1e-2) && (maxrel < 0.05);

  double checksum = 0.0;
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) checksum += h2d(OUT[i]);

  printf("OURS merlin_rvv_fp16_f32acc  M=%d N=%d K=%d\n", M, N, K);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("COS %d (x1e7)  REL_L2 %d (x1e7)  MAX_REL %d (x1e7)\n",
         (int)(cos * 1e7), (int)(rel_l2 * 1e7), (int)(maxrel * 1e7));
  printf("VERIFY %s errors=%d\n", pass ? "PASS" : "FAIL", pass ? 0 : 1);
  printf("CYCLES %lu\n", cycles);
  printf("CYCLES_FULL %lu\n", cycles_full);
  printf("FILL_CYCLES %lu\n", fill_cycles);
  printf("INSTRET %lu\n", instrs);
  printf("INSTRET_FULL %lu\n", instrs_full);
  return 0;
}
