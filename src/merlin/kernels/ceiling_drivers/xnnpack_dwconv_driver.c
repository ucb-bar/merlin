// CEILING driver: XNNPACK RVV f32 3x3 DEPTHWISE conv microkernel
// (xnn_f32_dwconv_ukernel_9p8vc__rvv). DEPTHWISE (one filter per channel) is a
// DIFFERENT op from a regular conv2d — see cross_framework_ops_k1.md: regular
// f32 conv on our side is im2col->GEMM (the GEMM ceiling), and XNNPACK's only
// f32 conv RVV kernel is this depthwise one, so we race ours-depthwise vs XNNPACK
// depthwise on a depthwise shape.
//
// Shape: OUT_H x OUT_W spatial, C channels, 3x3 kernel, stride 1, VALID padding
// (so IN_H=OUT_H+2, IN_W=OUT_W+2). The 9p kernel consumes a per-output-pixel
// indirection buffer of 9 input-row pointers; we build it for one output row at a
// time (output_width = OUT_W) and loop output rows. The weight buffer is packed
// [bias[C]] then 9 tap-panels of [C] (channel-major), the layout the kernel reads.
//
// inner_compute: indirection-buffer build + weight pack are OUTSIDE timing; only
// the kernel calls (one per output row) are timed.

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
#include "util.h"
#include "src/xnnpack/dwconv.h"

// ---- the expert microkernel, verbatim --------------------------------------
#include "f32-dwconv/gen/f32-dwconv-9p8vc-rvv.c"

#ifndef DW_OH
#define DW_OH 28
#endif
#ifndef DW_OW
#define DW_OW 28
#endif
#ifndef DW_C
#define DW_C 128
#endif
#define OH DW_OH
#define OW DW_OW
#define C  DW_C
#define IH (OH + 2)
#define IW (OW + 2)
#define KH 3
#define KW 3

// NHWC input/output (channels contiguous — the layout dwconv reads).
static float IN[IH * IW * C];
static float OUTbuf[OH * OW * C];
static float Ref[OH * OW * C];
static float Wf[KH * KW * C];     // per-(tap,channel) filter
static float biasf[C];
// packed weights: per channel-block of vlmax lanes -> [bias][9 taps], each vlmax wide.
// Sized for C rounded up to a generous vlmax bound (256 lanes) so any VLEN tail fits.
static float Wpack[(1 + KH * KW) * (C + 256)];
static float zero_row[C];         // dwconv "zero" sentinel (no padding here, unused taps)

// indirection buffer: for each output pixel, KH*KW input-row base pointers.
static const float* indir[OW * KH * KW];

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  // ---- init input / weights / bias -----------------------------------------
  for (int i = 0; i < IH * IW * C; i++)
    IN[i] = (float)((i % 13) - 6) * 0.0625f;
  for (int t = 0; t < KH * KW; t++)
    for (int c = 0; c < C; c++)
      Wf[t * C + c] = (float)(((t * 5 + c * 3) % 17) - 8) * 0.03125f;
  for (int c = 0; c < C; c++) { biasf[c] = (float)((c % 5) - 2) * 0.25f; zero_row[c] = 0.0f; }

  // ---- scalar reference: depthwise conv, stride 1, VALID --------------------
  for (int oy = 0; oy < OH; oy++)
    for (int ox = 0; ox < OW; ox++)
      for (int c = 0; c < C; c++) {
        float acc = biasf[c];
        for (int ky = 0; ky < KH; ky++)
          for (int kx = 0; kx < KW; kx++) {
            int iy = oy + ky, ix = ox + kx;
            acc += IN[(iy * IW + ix) * C + c] * Wf[(ky * KW + kx) * C + c];
          }
        Ref[(oy * OW + ox) * C + c] = acc;
      }

  // ---- pack weights: the kernel reads PER CHANNEL-BLOCK of vlmax (=vsetvlmax_e32m8)
  // lanes: [bias[blk]] [tap0[blk]] ... [tap8[blk]], advancing w by vlmax after each
  // panel. So for C > vlmax the panels are interleaved per block, NOT one big [bias C].
  {
    const size_t vlmax = __riscv_vsetvlmax_e32m8();
    float* p = Wpack;
    for (size_t c0 = 0; c0 < (size_t)C; c0 += vlmax) {
      size_t blk = (c0 + vlmax <= (size_t)C) ? vlmax : ((size_t)C - c0);
      // bias panel (vlmax wide; tail lanes unused by the kernel via vl<blk)
      for (size_t l = 0; l < vlmax; l++) p[l] = (l < blk) ? biasf[c0 + l] : 0.0f;
      p += vlmax;
      for (int t = 0; t < KH * KW; t++) {
        for (size_t l = 0; l < vlmax; l++) p[l] = (l < blk) ? Wf[t * C + (c0 + l)] : 0.0f;
        p += vlmax;
      }
    }
  }

  for (int i = 0; i < OH * OW * C; i++) OUTbuf[i] = 0.0f;

  // dwconv strides (bytes). The kernel advances `input` by input_stride per output
  // pixel and adds input_offset (we fold offset into the pointers, offset=0).
  const intptr_t input_stride = (intptr_t)(KH * KW) * sizeof(void*);  // 9 ptrs/pixel
  const size_t output_increment = 0;   // output advances by vl internally; rows contiguous
  const size_t input_offset = 0;
  const size_t input_pixel_stride = 0;
  struct xnn_f32_default_params params;

  // ---- TIMED region: one kernel call per output row -------------------------
  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  for (int oy = 0; oy < OH; oy++) {
    // build indirection for this output row: KH*KW row pointers per output x
    for (int ox = 0; ox < OW; ox++)
      for (int ky = 0; ky < KH; ky++)
        for (int kx = 0; kx < KW; kx++) {
          int iy = oy + ky, ix = ox + kx;
          indir[ox * (KH * KW) + ky * KW + kx] = &IN[(iy * IW + ix) * C];
        }
    xnn_f32_dwconv_ukernel_9p8vc__rvv(
        (size_t)C, (size_t)OW, indir, Wpack, &OUTbuf[oy * OW * C],
        input_stride, output_increment, input_offset, input_pixel_stride,
        zero_row, &params);
  }
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);
  unsigned long cycles = c1 - c0;
  unsigned long instrs = i1 - i0;

  // ---- verify ---------------------------------------------------------------
  int errors = 0;
  float maxabs = 0.0f;
  for (int i = 0; i < OH * OW * C; i++) {
    float d = OUTbuf[i] - Ref[i];
    if (d < 0) d = -d;
    if (d > maxabs) maxabs = d;
    if (d > 2e-3f) errors++;
  }
  double checksum = 0.0;
  for (int i = 0; i < OH * OW * C; i++) checksum += OUTbuf[i];

  printf("XNNPACK f32_dwconv_9p8vc__rvv  OH=%d OW=%d C=%d 3x3\n", OH, OW, C);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("O[0]=%d O[last]=%d (x1000)  maxabs_err=%d (x1e6)\n",
         (int)(OUTbuf[0] * 1000.0f), (int)(OUTbuf[OH*OW*C - 1] * 1000.0f), (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
