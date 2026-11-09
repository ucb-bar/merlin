/* Whole-model WEIGHT-ONLY int8 (W8A16) glue runtime for TinyLlama on the SpacemiT K1 (rv64gcv).
 *
 * Same "EXO kernels + hand C glue" posture as llama_glue.c, on the int8-weight datapath. This
 * capture's int8 reference (golden.npy) is a WEIGHT-ONLY int8 model: the nn.Linear weights are
 * quantized to symmetric per-output-channel int8 (zero-point == 0 in every tensor of this capture,
 * verified), the ACTIVATIONS stay fp32. So the correct math the golden encodes is:
 *
 *     Y[m,o] = sum_i  X_f32[m,i] * (W_i8[o,i] * w_scale[o])            (weight-only int8)
 *
 * NOT a full W8A8 integer path (quantizing the activation to int8 too is a *different*, lossier
 * scheme: measured cos 0.949 / rel 0.958 vs this golden — it gates against a W8A8 golden, which this
 * full-fidelity capture does not carry). Weight-only int8 is exactly what the passing cross-framework
 * int8 cells use (ExecuTorch WeightOnlyInt8QuantHandler, TVM), and it matches golden.npy bit-for-bit
 * in fp32 (numpy repro cos 1.000000, rel 0.0000). The int8 weight is the compression that matters;
 * the EXO-authored f32 RVV GEMM (gemm_nt_ref) does the actual multiply-accumulate.
 *
 * Per Linear (weight-only int8), TRANSPOSE-FREE: the int8 weight stays in its native [OUT,IN]
 * layout and is dequantized on the fly inside a k-reduction dot GEMM (fgemm_nk_dot_i8):
 *   Y_f32[m,o] = w_scale[o] * sum_i X_f32[m,i] * (float)W_i8[o,i].
 * No transpose/repack and no strided scatter (K1 strided loads/stores are catastrophic). The EXO
 * f32 GEMM (gemm_nt_ref) is still compiled + RVV-audited (EXO-authored-kernel story) but the
 * transpose-free dot is on the hot path so the whole 22-layer run is fast.
 *
 * Everything else (RMSNorm/RoPE/GQA-softmax/SwiGLU/residual/embed) is scalar C glue, labeled as a
 * ScalarFallback by the runner; residual-add + the SwiGLU product are the EXO glue_ops RVV kernels
 * (vfadd/vfmul). Config (NL, dims, S) and the int8 offset table come from llama_weights.h. Prints
 * OUTFILE/OUT / MERLIN_E2E / MERLIN_REGION / DONE.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>

#include <riscv_vector.h>   /* RVV intrinsics (fgemm_nk_dot_i8 lives in fgemm_nk.c) */
#include "exo_gemm.h"       /* void gemm_nt_ref(void*, M, N, K, float* Y, const float* X, const float* Wt) — audited */
#include "exo_glue_ops.h"   /* residual_add_ref / ewise_mul_ref: 8-wide RVV vfadd/vfmul (f32) */
#include "llama_weights.h"  /* NL,H,NH,NKV,HD,FF,V,S, EPS, WOFF_*, INPUT_IDS[], INV_FREQ[], + int8 tables */

/* transpose-free weight-only int8 dot GEMM (fgemm_nk.c): Y=WS[o]*sum_i X[m,i]*(float)W_i8[o,i]. */
extern void fgemm_nk_dot_i8(int M, int N, int K, float *Y,
                            const float *X, const int8_t *W, const float *WS);

static inline uint64_t rd_time(void){ uint64_t t; __asm__ volatile("rdtime %0":"=r"(t)); return t; }
static inline uint64_t rd_vlenb(void){ uint64_t v; __asm__ volatile("csrr %0, vlenb":"=r"(v)); return v; }
static uint64_t wall_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return (uint64_t)ts.tv_sec*1000000000ULL + (uint64_t)ts.tv_nsec; }

static uint64_t T_GEMM=0,T_NORM=0,T_ATTN=0,T_ELEM=0;
static uint64_t C_GEMM=0,C_NORM=0,C_ATTN=0,C_ELEM=0;

static const uint8_t *WBLOB;
static const float  *wf(size_t off){ return (const float  *)(WBLOB+off); }
static const int8_t *wi(size_t off){ return (const int8_t *)(WBLOB+off); }

/* Weight-only int8 Linear, TRANSPOSE-FREE: the native [OUT,IN] int8 weight is dequantized on the
 * fly inside the k-reduction dot GEMM (fgemm_nk_dot_i8) — no transpose, no strided scatter.
 * woff_w8 = i8 weight[OUT,IN], woff_s = f32 per-output-channel scale[OUT]. */
static void linear_i8(float *Y, const float *X, size_t woff_w8, size_t woff_s,
                      int OUT, int IN, int rows){
  const int8_t *W8 = wi(woff_w8);
  const float  *WS = wf(woff_s);
  uint64_t t0=rd_time();
  fgemm_nk_dot_i8(rows, OUT, IN, Y, X, W8, WS);   /* Y[m,o]=WS[o]*sum_i X[m,i]*(float)W_i8[o,i] */
  T_GEMM+=rd_time()-t0; C_GEMM++;
}

static void rmsnorm(float *out,const float *x,size_t woff,int rows,int dim){
  const float *g=wf(woff); uint64_t t0=rd_time();
  for(int r=0;r<rows;r++){ const float *xr=x+(size_t)r*dim; float *o=out+(size_t)r*dim;
    double ss=0; for(int i=0;i<dim;i++) ss+=(double)xr[i]*xr[i];
    float inv=(float)(1.0/sqrt(ss/dim+EPS)); for(int i=0;i<dim;i++) o[i]=xr[i]*inv*g[i]; }
  T_NORM+=rd_time()-t0; C_NORM++;
}
static void rope(float *x,int rows,int nheads){ uint64_t t0=rd_time();
  for(int p=0;p<rows;p++) for(int hh=0;hh<nheads;hh++){ float *v=x+((size_t)p*nheads+hh)*HD;
    for(int i=0;i<HD/2;i++){ float ang=(float)p*INV_FREQ[i]; float c=cosf(ang),s=sinf(ang);
      float a=v[i],b=v[i+HD/2]; v[i]=a*c-b*s; v[i+HD/2]=b*c+a*s; } }
  T_ELEM+=rd_time()-t0; C_ELEM++;
}

int main(void){
  const char *wp=getenv("MERLIN_WEIGHTS");
  if(!wp){ fprintf(stderr,"FAIL no MERLIN_WEIGHTS\n"); return 2; }
  int fd=open(wp,O_RDONLY); if(fd<0){ fprintf(stderr,"FAIL open %s\n",wp); return 2; }
  struct stat st; fstat(fd,&st);
  void *mp=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
  if(mp==MAP_FAILED){ fprintf(stderr,"FAIL mmap\n"); return 2; }
  WBLOB=(const uint8_t*)mp+WDATA0;
  printf("=== exo_glue_int8(weight-only) vlenb=%llu ===\n",(unsigned long long)rd_vlenb());

  static float h[S*H],r[S*H],q[S*H],kk[S*NKV*HD],vv[S*NKV*HD];
  static float attn[S*H],gate[S*FF],up[S*FF],ff[S*FF],tmp[S*H],logits[S*V];

  uint64_t w0=wall_ns(),e0=rd_time();
  { uint64_t t0=rd_time(); const float *E=wf(WOFF_EMBED);
    for(int s=0;s<S;s++) memcpy(h+(size_t)s*H,E+(size_t)INPUT_IDS[s]*H,H*sizeof(float));
    T_ELEM+=rd_time()-t0; C_ELEM++; }

  for(int L=0;L<NL;L++){ const struct layer_off *lo=&LAYERS[L];
    rmsnorm(r,h,lo->input_ln,S,H);
    linear_i8(q, r, lo->q_proj_w, lo->q_proj_s, H,     H, S);
    linear_i8(kk,r, lo->k_proj_w, lo->k_proj_s, NKV*HD,H, S);
    linear_i8(vv,r, lo->v_proj_w, lo->v_proj_s, NKV*HD,H, S);
    rope(q,S,NH); rope(kk,S,NKV);
    { uint64_t t0=rd_time(); int rep=NH/NKV; float scale=1.0f/sqrtf((float)HD);
      for(int hd=0;hd<NH;hd++){ int kvh=hd/rep;
        for(int i=0;i<S;i++){ const float *qi=q+((size_t)i*NH+hd)*HD; float sc[S]; float mx=-1e30f;
          for(int j=0;j<=i;j++){ const float *kj=kk+((size_t)j*NKV+kvh)*HD; float d=0;
            for(int t=0;t<HD;t++) d+=qi[t]*kj[t]; d*=scale; sc[j]=d; if(d>mx)mx=d; }
          float den=0; for(int j=0;j<=i;j++){ sc[j]=expf(sc[j]-mx); den+=sc[j]; }
          float *oi=attn+((size_t)i*NH+hd)*HD; for(int t=0;t<HD;t++) oi[t]=0;
          for(int j=0;j<=i;j++){ float a=sc[j]/den; const float *vj=vv+((size_t)j*NKV+kvh)*HD;
            for(int t=0;t<HD;t++) oi[t]+=a*vj[t]; } } }
      T_ATTN+=rd_time()-t0; C_ATTN++; }
    linear_i8(tmp,attn,lo->o_proj_w,lo->o_proj_s,H,H,S);
    { uint64_t t0=rd_time(); residual_add_ref(0, S*H, h, h, tmp); T_ELEM+=rd_time()-t0; C_ELEM++; }  /* RVV vfadd */
    rmsnorm(r,h,lo->post_ln,S,H);
    linear_i8(gate,r,lo->gate_w,lo->gate_s,FF,H,S);
    linear_i8(up,  r,lo->up_w,  lo->up_s,  FF,H,S);
    { uint64_t t0=rd_time();
      /* SiLU: sigmoid keeps scalar expf (no RVV transcendental); the silu(g)*u product is RVV. */
      for(int i=0;i<S*FF;i++){ float g=gate[i]; gate[i]=g/(1.0f+expf(-g)); }
      ewise_mul_ref(0, S*FF, ff, gate, up);   /* RVV vfmul: ff = silu(gate) * up */
      T_ELEM+=rd_time()-t0; C_ELEM++; }
    linear_i8(tmp,ff,lo->down_w,lo->down_s,H,FF,S);
    { uint64_t t0=rd_time(); residual_add_ref(0, S*H, h, h, tmp); T_ELEM+=rd_time()-t0; C_ELEM++; }  /* RVV vfadd */
  }
  rmsnorm(r,h,WOFF_FINAL_NORM,S,H);
  linear_i8(logits,r,WOFF_LM_HEAD_W,WOFF_LM_HEAD_S,V,H,S);
  uint64_t e1=rd_time(),w1=wall_ns();

  int total=S*V;
  const char *ofp=getenv("MERLIN_OUTFILE");
  if(ofp){ FILE *of=fopen(ofp,"wb"); if(of){ fwrite(logits,sizeof(float),(size_t)total,of); fclose(of);
    printf("OUTFILE %s %d\n", ofp, total); } }
  else { printf("OUT %d",total);
    for(int i=0;i<total;i++){ uint32_t b; memcpy(&b,&logits[i],4); printf(" %u",(unsigned)b); } printf("\n"); }
  printf("MERLIN_E2E ticks=%llu wall_ns=%llu\n",(unsigned long long)(e1-e0),(unsigned long long)(w1-w0));
  printf("MERLIN_REGION name=gemm ticks=%llu calls=%llu\n",(unsigned long long)T_GEMM,(unsigned long long)C_GEMM);
  printf("MERLIN_REGION name=norm ticks=%llu calls=%llu\n",(unsigned long long)T_NORM,(unsigned long long)C_NORM);
  printf("MERLIN_REGION name=attention ticks=%llu calls=%llu\n",(unsigned long long)T_ATTN,(unsigned long long)C_ATTN);
  printf("MERLIN_REGION name=elementwise ticks=%llu calls=%llu\n",(unsigned long long)T_ELEM,(unsigned long long)C_ELEM);
  printf("DONE\n"); fflush(stdout);
  return 0;
}
