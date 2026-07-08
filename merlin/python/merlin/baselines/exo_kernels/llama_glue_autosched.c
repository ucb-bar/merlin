/* Whole-model fp32 glue runtime for a Llama-family decoder on the SpacemiT K1 (rv64gcv), using the
 * EXO-AUTOSCHEDULED transpose-free dot GEMM (autosched.py -> fdot_nk_ref) as the per-Linear kernel.
 *
 * Identical whole-model forward to llama_glue.c, but the nn.Linear GEMM is the kernel EXO's OWN
 * `vectorize` auto-op generated (8-wide RVV partial-sum accumulator + contiguous vle + vfmacc.vv +
 * vfredusum horizontal reduce) rather than the hand-scheduled broadcast `gemm_nt_ref`. Because the
 * autoscheduled kernel is a transpose-free k-reduction dot (Y[m,o]=sum_k X[m,k]*W[o,k], weight kept
 * native [OUT,IN]), there is NO transpose pre-pass (the fp32 glue's WT_SCRATCH scatter is gone).
 *
 * Everything else (RMSNorm/RoPE/GQA-softmax/SwiGLU/residual/embed) is scalar C glue, labeled as a
 * ScalarFallback by the runner. Config + offset table come from llama_weights.h. Prints OUTFILE/OUT
 * / MERLIN_E2E / MERLIN_REGION / DONE — same profile brackets as the other arms.
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

#include "autosched_dot.h"  /* void fdot_nk_ref(void*, M, N, K, float* Y, const float* X, const float* Wf) — EXO-autoscheduled */
#include "llama_weights.h"  /* NL,H,NH,NKV,HD,FF,V,S, EPS, WOFF_*, INPUT_IDS[], INV_FREQ[] */

static inline uint64_t rd_time(void){ uint64_t t; __asm__ volatile("rdtime %0":"=r"(t)); return t; }
static inline uint64_t rd_vlenb(void){ uint64_t v; __asm__ volatile("csrr %0, vlenb":"=r"(v)); return v; }
static uint64_t wall_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return (uint64_t)ts.tv_sec*1000000000ULL + (uint64_t)ts.tv_nsec; }

static uint64_t T_GEMM=0,T_NORM=0,T_ATTN=0,T_ELEM=0;
static uint64_t C_GEMM=0,C_NORM=0,C_ATTN=0,C_ELEM=0;

static const uint8_t *WBLOB;
static const float *wf(size_t off){ return (const float *)(WBLOB+off); }

/* Transpose-free Linear via the EXO-autoscheduled dot: Y[m,o]=sum_k X[m,k]*W[o,k] (W native [OUT,IN]). */
static void linear(float *Y, const float *X, size_t woff, int OUT, int IN, int rows){
  const float *W = wf(woff);
  uint64_t t0=rd_time();
  fdot_nk_ref(NULL, rows, OUT, IN, Y, X, W);   /* EXO-autoscheduled RVV dot */
  T_GEMM += rd_time()-t0; C_GEMM++;
}

static void rmsnorm(float *out, const float *x, size_t woff, int rows, int dim){
  const float *g=wf(woff); uint64_t t0=rd_time();
  for(int r=0;r<rows;r++){ const float *xr=x+(size_t)r*dim; float *o=out+(size_t)r*dim;
    double ss=0; for(int i=0;i<dim;i++) ss+=(double)xr[i]*xr[i];
    float inv=(float)(1.0/sqrt(ss/dim+EPS)); for(int i=0;i<dim;i++) o[i]=xr[i]*inv*g[i]; }
  T_NORM+=rd_time()-t0; C_NORM++;
}
static void rope(float *x, int rows, int nheads){ uint64_t t0=rd_time();
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
  printf("=== exo_glue_autosched vlenb=%llu ===\n",(unsigned long long)rd_vlenb());

  static float h[S*H],r[S*H],q[S*H],kk[S*NKV*HD],vv[S*NKV*HD];
  static float attn[S*H],gate[S*FF],up[S*FF],ff[S*FF],tmp[S*H],logits[S*V];

  uint64_t w0=wall_ns(),e0=rd_time();
  { uint64_t t0=rd_time(); const float *E=wf(WOFF_EMBED);
    for(int s=0;s<S;s++) memcpy(h+(size_t)s*H,E+(size_t)INPUT_IDS[s]*H,H*sizeof(float));
    T_ELEM+=rd_time()-t0; C_ELEM++; }

  for(int L=0;L<NL;L++){ const struct layer_off *lo=&LAYERS[L];
    rmsnorm(r,h,lo->input_ln,S,H);
    linear(q, r, lo->q_proj, H,     H, S);
    linear(kk,r, lo->k_proj, NKV*HD,H, S);
    linear(vv,r, lo->v_proj, NKV*HD,H, S);
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
    linear(tmp,attn,lo->o_proj,H,H,S);
    { uint64_t t0=rd_time(); for(int i=0;i<S*H;i++) h[i]+=tmp[i]; T_ELEM+=rd_time()-t0; C_ELEM++; }
    rmsnorm(r,h,lo->post_ln,S,H);
    linear(gate,r,lo->gate_proj,FF,H,S);
    linear(up,  r,lo->up_proj,  FF,H,S);
    { uint64_t t0=rd_time();
      for(int i=0;i<S*FF;i++){ float g=gate[i]; ff[i]=(g/(1.0f+expf(-g)))*up[i]; }
      T_ELEM+=rd_time()-t0; C_ELEM++; }
    linear(tmp,ff,lo->down_proj,H,FF,S);
    { uint64_t t0=rd_time(); for(int i=0;i<S*H;i++) h[i]+=tmp[i]; T_ELEM+=rd_time()-t0; C_ELEM++; }
  }
  rmsnorm(r,h,WOFF_FINAL_NORM,S,H);
  linear(logits,r,WOFF_LM_HEAD,V,H,S);
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
