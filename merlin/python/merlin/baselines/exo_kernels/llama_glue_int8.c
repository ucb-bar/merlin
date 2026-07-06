/* Whole-model int8 (W8A8) glue runtime for TinyLlama on the SpacemiT K1 (rv64gcv, glibc Linux).
 *
 * Same "EXO kernels + hand C glue" posture as llama_glue.c, on the INTEGER datapath: the dominant
 * op (nn.Linear) is the EXO int8 RVV GEMM (exo_kernels/igemm.py -> igemm_nt_ref, a vwmacc
 * i16xi16->i32 widening MAC at VLEN=256). Per Linear:
 *   1. quantize the activation row per-token to i8 (symmetric, per-row max-abs scale), stored i16;
 *   2. load the i8 weight, transpose to [K,N] and sign-extend to i16 (glue pre-pass, off timing);
 *   3. igemm_nt_ref: acc_i32[m,o] = sum_i A_i16[m,i]*Wt_i16[i,o]  (EXO RVV vwmacc);
 *   4. requantize: Y_f32[m,o] = a_scale[m] * w_scale[o] * acc_i32[m,o]   (scalar glue).
 * Weights are symmetric per-output-channel int8 (zero-point = 0 in this capture).
 *
 * Everything else (RMSNorm/RoPE/GQA-softmax/SwiGLU/residual/embed) is scalar C glue, labeled as a
 * ScalarFallback by the runner. Config (NL, dims, S) and the int8 offset table come from
 * llama_weights.h (emitted by exo.py). Prints OUT / MERLIN_E2E / MERLIN_REGION / DONE.
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

#include "exo_igemm.h"      /* void igemm_nt_ref(void*, M, N, K, int32_t* Y, const uint16_t* X, const uint16_t* Wt) */
#include "llama_weights.h"  /* NL,H,NH,NKV,HD,FF,V,S, EPS, WOFF_*, INPUT_IDS[], INV_FREQ[], + int8 tables */

static inline uint64_t rd_time(void){ uint64_t t; __asm__ volatile("rdtime %0":"=r"(t)); return t; }
static inline uint64_t rd_vlenb(void){ uint64_t v; __asm__ volatile("csrr %0, vlenb":"=r"(v)); return v; }
static uint64_t wall_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return (uint64_t)ts.tv_sec*1000000000ULL + (uint64_t)ts.tv_nsec; }

static uint64_t T_GEMM=0,T_NORM=0,T_ATTN=0,T_ELEM=0,T_QUANT=0;
static uint64_t C_GEMM=0,C_NORM=0,C_ATTN=0,C_ELEM=0,C_QUANT=0;

static const uint8_t *WBLOB;
static const float  *wf(size_t off){ return (const float  *)(WBLOB+off); }
static const int8_t *wi(size_t off){ return (const int8_t *)(WBLOB+off); }

/* scratch: transposed+widened i16 weight [K,N]; quantized i16 activation [S,K]; i32 acc [S,N]. */
static int16_t *WT16; static int16_t *X16; static int32_t *ACC;

/* int8 W8A8 Linear via the EXO vwmacc kernel. woff_w8=i8 weight[OUT,IN], woff_s=f32 scale[OUT]. */
static void linear_i8(float *Y, const float *X, size_t woff_w8, size_t woff_s,
                      int OUT, int IN, int rows){
  const int8_t *W8 = wi(woff_w8);
  const float  *WS = wf(woff_s);
  /* 1. per-token activation quant -> i16 (holds i8 range) + per-row scale */
  uint64_t tq=rd_time();
  static float ASCALE[64];          /* rows <= 64 for these workloads (S=8) */
  for(int r=0;r<rows;r++){
    const float *xr=X+(size_t)r*IN; float amax=0.0f;
    for(int i=0;i<IN;i++){ float a=fabsf(xr[i]); if(a>amax)amax=a; }
    float sc = amax>0.0f ? amax/127.0f : 1.0f; ASCALE[r]=sc;
    int16_t *xo=X16+(size_t)r*IN; float inv=1.0f/sc;
    for(int i=0;i<IN;i++){ int q=(int)lrintf(xr[i]*inv); if(q>127)q=127; if(q<-128)q=-128; xo[i]=(int16_t)q; }
  }
  /* 2. transpose + widen weight i8[OUT,IN] -> i16[IN,OUT] */
  for(int o=0;o<OUT;o++){ const int8_t *wr=W8+(size_t)o*IN; for(int i=0;i<IN;i++) WT16[(size_t)i*OUT+o]=(int16_t)wr[i]; }
  T_QUANT+=rd_time()-tq; C_QUANT++;
  /* 3. EXO RVV int8 GEMM: ACC_i32 = X16 @ WT16 */
  uint64_t t0=rd_time();
  igemm_nt_ref(NULL, rows, OUT, IN, ACC, (const uint16_t*)X16, (const uint16_t*)WT16);
  T_GEMM+=rd_time()-t0; C_GEMM++;
  /* 4. requant: Y = a_scale[r]*w_scale[o]*acc (scalar glue) */
  uint64_t tr=rd_time();
  for(int r=0;r<rows;r++){ float as=ASCALE[r]; const int32_t *ar=ACC+(size_t)r*OUT; float *yr=Y+(size_t)r*OUT;
    for(int o=0;o<OUT;o++) yr[o]=as*WS[o]*(float)ar[o]; }
  T_ELEM+=rd_time()-tr; C_ELEM++;
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
  printf("=== exo_glue_int8 vlenb=%llu ===\n",(unsigned long long)rd_vlenb());

  static float h[S*H],r[S*H],q[S*H],kk[S*NKV*HD],vv[S*NKV*HD];
  static float attn[S*H],gate[S*FF],up[S*FF],ff[S*FF],tmp[S*H],logits[S*V];
  WT16=(int16_t*)malloc((size_t)V*H*sizeof(int16_t));
  X16 =(int16_t*)malloc((size_t)S*(FF>H?FF:H)*sizeof(int16_t));
  ACC =(int32_t*)malloc((size_t)S*V*sizeof(int32_t));
  if(!WT16||!X16||!ACC){ fprintf(stderr,"FAIL scratch\n"); return 2; }

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
    { uint64_t t0=rd_time(); for(int i=0;i<S*H;i++) h[i]+=tmp[i]; T_ELEM+=rd_time()-t0; C_ELEM++; }
    rmsnorm(r,h,lo->post_ln,S,H);
    linear_i8(gate,r,lo->gate_w,lo->gate_s,FF,H,S);
    linear_i8(up,  r,lo->up_w,  lo->up_s,  FF,H,S);
    { uint64_t t0=rd_time(); for(int i=0;i<S*FF;i++){ float g=gate[i]; ff[i]=(g/(1.0f+expf(-g)))*up[i]; }
      T_ELEM+=rd_time()-t0; C_ELEM++; }
    linear_i8(tmp,ff,lo->down_w,lo->down_s,H,FF,S);
    { uint64_t t0=rd_time(); for(int i=0;i<S*H;i++) h[i]+=tmp[i]; T_ELEM+=rd_time()-t0; C_ELEM++; }
  }
  rmsnorm(r,h,WOFF_FINAL_NORM,S,H);
  linear_i8(logits,r,WOFF_LM_HEAD_W,WOFF_LM_HEAD_S,V,H,S);
  uint64_t e1=rd_time(),w1=wall_ns();

  int total=S*V; printf("OUT %d",total);
  for(int i=0;i<total;i++){ uint32_t b; memcpy(&b,&logits[i],4); printf(" %u",(unsigned)b); } printf("\n");
  printf("MERLIN_E2E ticks=%llu wall_ns=%llu\n",(unsigned long long)(e1-e0),(unsigned long long)(w1-w0));
  printf("MERLIN_REGION name=gemm ticks=%llu calls=%llu\n",(unsigned long long)T_GEMM,(unsigned long long)C_GEMM);
  printf("MERLIN_REGION name=norm ticks=%llu calls=%llu\n",(unsigned long long)T_NORM,(unsigned long long)C_NORM);
  printf("MERLIN_REGION name=attention ticks=%llu calls=%llu\n",(unsigned long long)T_ATTN,(unsigned long long)C_ATTN);
  printf("MERLIN_REGION name=elementwise ticks=%llu calls=%llu\n",(unsigned long long)T_ELEM,(unsigned long long)C_ELEM);
  printf("MERLIN_REGION name=other ticks=%llu calls=%llu\n",(unsigned long long)T_QUANT,(unsigned long long)C_QUANT);
  printf("DONE\n"); fflush(stdout);
  return 0;
}
