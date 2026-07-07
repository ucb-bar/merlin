/* Whole-model glue runtime for TinyLlama on the SpacemiT K1 (rv64gcv, glibc Linux).
 *
 * "Whole-model" here = EXO-generated RVV kernels + a hand C glue runtime. EXO is a kernel DSL +
 * scheduler, NOT a whole-model compiler; so we generate the dominant op (the nn.Linear GEMM) as an
 * EXO RVV kernel (see exo_kernels/gemm.py -> gemm_nt_ref) and drive the full transformer forward
 * from this C harness, calling that kernel for every Linear. Everything else (RMSNorm, RoPE, GQA
 * softmax attention, SwiGLU/SiLU, residual adds) is SCALAR C glue — labeled honestly as a scalar
 * fallback (no EXO RVV kernel), NOT hidden.
 *
 * Weights are mmap'd from the capture's weights.safetensors blob using a generated offset table
 * (llama_weights.h, emitted by exo.py from the safetensors header). Inputs (8 token ids) and the
 * generated config also come from llama_weights.h. Prints:
 *   OUT <k> <bits...>            -- first k output logits (float bit patterns), for correctness
 *   MERLIN_E2E ticks=.. wall_ns=..
 *   MERLIN_REGION name=gemm|norm|attention|elementwise ticks=.. calls=..
 *   DONE
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

#include "exo_gemm.h"       /* void gemm_nt_ref(void*, M, N, K, float* Y, const float* X, const float* Wt) */
#include "llama_weights.h"  /* NL,H,NH,NKV,HD,FF,V,S, EPS, THETA, WOFF_*, INPUT_IDS[], INV_FREQ[] */

/* ---- timing (rdtime, 24 MHz platform timer -> est core cycles in profile.py) ---- */
static inline uint64_t rd_time(void){ uint64_t t; __asm__ volatile("rdtime %0":"=r"(t)); return t; }
static inline uint64_t rd_vlenb(void){ uint64_t v; __asm__ volatile("csrr %0, vlenb":"=r"(v)); return v; }
static uint64_t wall_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return (uint64_t)ts.tv_sec*1000000000ULL + (uint64_t)ts.tv_nsec; }

/* region accumulators */
static uint64_t T_GEMM=0, T_NORM=0, T_ATTN=0, T_ELEM=0;
static uint64_t C_GEMM=0, C_NORM=0, C_ATTN=0, C_ELEM=0;

static const uint8_t *WBLOB;   /* mmap'd safetensors, base of tensor data */

static const float *w(size_t off){ return (const float *)(WBLOB + off); }

/* transpose W[out,in] -> Wt[in,out] into a scratch buffer (glue pre-pack, NOT timed as gemm). */
static void transpose(const float *Wsrc, float *Wt, int OUT, int IN){
  for(int o=0;o<OUT;o++) for(int i=0;i<IN;i++) Wt[i*OUT+o]=Wsrc[o*IN+i];
}

/* Y[S,OUT] = X[S,IN] @ W[OUT,IN]^T, delegating to the EXO RVV kernel (X @ Wt, Wt=[IN,OUT]). */
static float *WT_SCRATCH; /* max IN*OUT reused; sized to the biggest linear (V*H) at startup */
static void linear(float *Y, const float *X, size_t woff, int OUT, int IN, int rows){
  const float *Wsrc = w(woff);
  transpose(Wsrc, WT_SCRATCH, OUT, IN);
  uint64_t t0=rd_time();
  gemm_nt_ref(NULL, rows, OUT, IN, Y, X, WT_SCRATCH);   /* EXO RVV GEMM */
  T_GEMM += rd_time()-t0; C_GEMM++;
}

static void rmsnorm(float *out, const float *x, size_t woff, int rows, int dim){
  const float *g = w(woff);
  uint64_t t0=rd_time();
  for(int r=0;r<rows;r++){
    const float *xr=x+(size_t)r*dim; float *o=out+(size_t)r*dim;
    double ss=0; for(int i=0;i<dim;i++) ss+=(double)xr[i]*xr[i];
    float inv=(float)(1.0/sqrt(ss/dim + EPS));
    for(int i=0;i<dim;i++) o[i]=xr[i]*inv*g[i];
  }
  T_NORM += rd_time()-t0; C_NORM++;
}

static void rope(float *x, int rows, int nheads){ /* x:[rows,nheads,HD] */
  uint64_t t0=rd_time();
  for(int p=0;p<rows;p++) for(int hh=0;hh<nheads;hh++){
    float *v=x+((size_t)p*nheads+hh)*HD;
    for(int i=0;i<HD/2;i++){
      float ang=(float)p*INV_FREQ[i];
      float c=cosf(ang), s=sinf(ang);
      float a=v[i], b=v[i+HD/2];
      v[i]=a*c - b*s; v[i+HD/2]=b*c + a*s;
    }
  }
  T_ELEM += rd_time()-t0; C_ELEM++;
}

int main(void){
  /* mmap the weights blob (MERLIN_WEIGHTS env or default remote path). */
  const char *wp=getenv("MERLIN_WEIGHTS");
  if(!wp){ fprintf(stderr,"FAIL no MERLIN_WEIGHTS\n"); return 2; }
  int fd=open(wp,O_RDONLY); if(fd<0){ fprintf(stderr,"FAIL open %s\n",wp); return 2; }
  struct stat st; fstat(fd,&st);
  void *mp=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
  if(mp==MAP_FAILED){ fprintf(stderr,"FAIL mmap\n"); return 2; }
  WBLOB=(const uint8_t*)mp + WDATA0;   /* skip the safetensors JSON header */

  printf("=== exo_glue vlenb=%llu ===\n",(unsigned long long)rd_vlenb());

  static float h[S*H], r[S*H], q[S*H], kk[S*NKV*HD], vv[S*NKV*HD];
  static float attn[S*H], gate[S*FF], up[S*FF], ff[S*FF], tmp[S*H];
  static float logits[S*V];
  /* WT_SCRATCH holds a transposed weight [IN,OUT]; size it to the LARGEST Linear = max(IN*OUT) over
   * {lm_head V*H, mlp down_proj FF*H, gate/up FF*H, qkvo H*H}. For a small vocab, FF*H can exceed
   * V*H (small_llama: FF*H=44032 > V*H=32768), so max over both — else down_proj overflows. */
  size_t wt_max=(size_t)V*H; if((size_t)FF*H>wt_max) wt_max=(size_t)FF*H;
  WT_SCRATCH=(float*)malloc(wt_max*sizeof(float));
  if(!WT_SCRATCH){ fprintf(stderr,"FAIL scratch\n"); return 2; }

  uint64_t w0=wall_ns(), e0=rd_time();

  /* embed_tokens: gather rows (elementwise/gather glue). */
  { uint64_t t0=rd_time(); const float *E=w(WOFF_EMBED);
    for(int s=0;s<S;s++) memcpy(h+(size_t)s*H, E+(size_t)INPUT_IDS[s]*H, H*sizeof(float));
    T_ELEM+=rd_time()-t0; C_ELEM++; }

  for(int L=0;L<NL;L++){
    const struct layer_off *lo=&LAYERS[L];
    rmsnorm(r,h,lo->input_ln,S,H);
    linear(q, r, lo->q_proj, H,    H, S);      /* [S,H]  */
    linear(kk,r, lo->k_proj, NKV*HD,H, S);     /* [S,NKV*HD] */
    linear(vv,r, lo->v_proj, NKV*HD,H, S);
    rope(q, S, NH);
    rope(kk,S, NKV);
    /* GQA causal softmax attention -> attn[S,H] (scalar glue). */
    { uint64_t t0=rd_time(); int rep=NH/NKV; float scale=1.0f/sqrtf((float)HD);
      for(int hd=0;hd<NH;hd++){ int kvh=hd/rep;
        for(int i=0;i<S;i++){
          const float *qi=q+((size_t)i*NH+hd)*HD;
          float sc[S]; float mx=-1e30f;
          for(int j=0;j<=i;j++){ const float *kj=kk+((size_t)j*NKV+kvh)*HD;
            float d=0; for(int t=0;t<HD;t++) d+=qi[t]*kj[t]; d*=scale; sc[j]=d; if(d>mx)mx=d; }
          float den=0; for(int j=0;j<=i;j++){ sc[j]=expf(sc[j]-mx); den+=sc[j]; }
          float *oi=attn+((size_t)i*NH+hd)*HD;
          for(int t=0;t<HD;t++) oi[t]=0;
          for(int j=0;j<=i;j++){ float a=sc[j]/den; const float *vj=vv+((size_t)j*NKV+kvh)*HD;
            for(int t=0;t<HD;t++) oi[t]+=a*vj[t]; }
        }
      }
      T_ATTN+=rd_time()-t0; C_ATTN++; }
    linear(tmp, attn, lo->o_proj, H, H, S);
    { uint64_t t0=rd_time(); for(int i=0;i<S*H;i++) h[i]+=tmp[i]; T_ELEM+=rd_time()-t0; C_ELEM++; }
    rmsnorm(r,h,lo->post_ln,S,H);
    linear(gate,r,lo->gate_proj, FF,H, S);
    linear(up,  r,lo->up_proj,   FF,H, S);
    { uint64_t t0=rd_time();
      for(int i=0;i<S*FF;i++){ float g=gate[i]; ff[i]=(g/(1.0f+expf(-g)))*up[i]; }
      T_ELEM+=rd_time()-t0; C_ELEM++; }
    linear(tmp, ff, lo->down_proj, H, FF, S);
    { uint64_t t0=rd_time(); for(int i=0;i<S*H;i++) h[i]+=tmp[i]; T_ELEM+=rd_time()-t0; C_ELEM++; }
  }
  rmsnorm(r,h,WOFF_FINAL_NORM,S,H);
  linear(logits, r, WOFF_LM_HEAD, V, H, S);

  uint64_t e1=rd_time(), w1=wall_ns();

  /* Write the full S*V logits to a binary file (MERLIN_OUTFILE) — the runner scp's it back and
   * cos/rel-gates vs golden.npy. Streaming ~2MB of ASCII over a contended SSH link truncates when
   * the connection drops; a small binary file is robust. Print OUTFILE + OUT (count) markers. */
  int total=S*V;
  const char *ofp=getenv("MERLIN_OUTFILE");
  if(ofp){ FILE *of=fopen(ofp,"wb"); if(of){ fwrite(logits,sizeof(float),(size_t)total,of); fclose(of);
    printf("OUTFILE %s %d\n", ofp, total); } }
  else { /* fallback: stream (small V only) */
    printf("OUT %d", total);
    for(int i=0;i<total;i++){ uint32_t b; memcpy(&b,&logits[i],4); printf(" %u",(unsigned)b); }
    printf("\n"); }

  printf("MERLIN_E2E ticks=%llu wall_ns=%llu\n",(unsigned long long)(e1-e0),(unsigned long long)(w1-w0));
  printf("MERLIN_REGION name=gemm ticks=%llu calls=%llu\n",(unsigned long long)T_GEMM,(unsigned long long)C_GEMM);
  printf("MERLIN_REGION name=norm ticks=%llu calls=%llu\n",(unsigned long long)T_NORM,(unsigned long long)C_NORM);
  printf("MERLIN_REGION name=attention ticks=%llu calls=%llu\n",(unsigned long long)T_ATTN,(unsigned long long)C_ATTN);
  printf("MERLIN_REGION name=elementwise ticks=%llu calls=%llu\n",(unsigned long long)T_ELEM,(unsigned long long)C_ELEM);
  printf("DONE\n");
  fflush(stdout);
  return 0;
}
