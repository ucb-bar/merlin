#include <assert.h>
#include <fenv.h>
#include <stdio.h>
#include <string.h>
#include "provider.c"

static unsigned long long checks;
/* Independent producer scan supplies the same exact immutable word facts.
 * The selected consumer itself is forbidden from repeating this scan. */
static void check(const float *source, size_t n, unsigned digits, int endpoints) {
  uint32_t maximum=0,minimum=UINT32_C(0x7f800000);
  float lo[192],hi[192];
  for(size_t i=0;i<n;i++) {
    uint32_t raw;memcpy(&raw,source+i,4);
    uint32_t magnitude=raw&UINT32_C(0x7fffffff);
    assert(!(raw&UINT32_C(0xffff)) && magnitude<UINT32_C(0x7f800000));
    if(magnitude>maximum)maximum=magnitude;
    if(magnitude<minimum)minimum=magnitude;
    lo[i]=source[i];hi[i]=source[i];
    if(endpoints==2 && i%3==0)hi[i]=merlin_fma_next_up_f32(source[i]);
  }
  double rd0[384],rd1[384];int8_t p0[1152],p1[1152];
  memset(rd0,0xa5,sizeof(rd0));memset(rd1,0xa5,sizeof(rd1));
  memset(p0,0x5a,sizeof(p0));memset(p1,0x5a,sizeof(p1));
  float step0=19,step1=19;unsigned char f0=3,f1=3;
  merlin_fma_bound environment=merlin_fma_bound_begin();
  int a=merlin_bf16_radix_row_widen(&environment,source,n,1,rd0,2,p0,384,2,digits,&step0,endpoints?lo:0,endpoints?hi:0,&f0);
  int b=merlin_bf16_radix_row_widen_produced(&environment,source,n,1,rd1,2,p1,384,2,digits,&step1,endpoints?lo:0,endpoints?hi:0,&f1,maximum,minimum);
  assert(a==b && f0==f1 && !memcmp(&step0,&step1,4));
  assert(!memcmp(rd0,rd1,sizeof(rd0)) && !memcmp(p0,p1,sizeof(p0)));
  checks++;
}

int main(void) {
  float row[192];
  /* Exhaustive signed BF16 words, including both zeros and subnormals. */
  for(uint32_t word=0;word<65536;word++) {
    uint32_t raw=word<<16;
    if((raw&UINT32_C(0x7fffffff))>=UINT32_C(0x7f800000))continue;
    memcpy(row,&raw,4);
    for(unsigned digits=1;digits<=3;digits++)check(row,1,digits,word%3);
  }
  uint32_t state=UINT32_C(0x312fa653);
  const size_t lengths[]={1,17,64,192};
  for(unsigned sample=0;sample<256;sample++) {
    for(size_t i=0;i<192;i++) {
      state=state*UINT32_C(1664525)+UINT32_C(1013904223);
      uint32_t word=(state>>16)&UINT32_C(0xffff);
      if((word&UINT32_C(0x7f80))==UINT32_C(0x7f80))word&=UINT32_C(0xff7f);
      uint32_t raw=word<<16;memcpy(row+i,&raw,4);
    }
    for(unsigned length=0;length<4;length++)for(unsigned digits=1;digits<=3;digits++)for(int e=0;e<3;e++)check(row,lengths[length],digits,e);
  }
  /* Original eligibility refusal survives every non-RNE environment. */
  const int modes[]={FE_TOWARDZERO,FE_DOWNWARD,FE_UPWARD};
  row[0]=1;row[1]=-0.0f;row[2]=0x1p-130f;
  for(unsigned mode=0;mode<3;mode++) {
    assert(!fesetround(modes[mode]));check(row,3,3,0);
  }
  assert(!fesetround(FE_TONEAREST));
  puts("PRODUCED_RADIX_WORD_EQUIVALENCE PASS");
  printf("CHECKS %llu\n",checks);
  return 0;
}
