/* Ordered FP8 product / BF16 accumulator reference for one live row; padded rows stay exact +zero. */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
static float fp8(uint8_t b) {
  unsigned e=(b>>3)&15, f=b&7;
  if (!e) return 0;
  return ldexpf((float)(8+f), (int)e-10)*(b&128?-1.0f:1.0f);
}
static uint16_t bf(float f) {
  union {float f; uint32_t u;} v={.f=f};
  return (uint16_t)((v.u+0x7fff+((v.u>>16)&1))>>16);
}
void tiny_reference(const uint8_t *a,const uint8_t *b,size_t k,size_t n,int scale,uint16_t *out) {
  for(size_t j=0;j<n;j++) {
    uint16_t acc=0;
    for(size_t q=0;q<k;q++) {
      float p=fp8(a[(q/32)*1024+q%32])*fp8(b[(j/32)*(k/32)*1024+(q/32)*1024+(q%32)*32+j%32]);
      if(p) { union {float f; uint32_t u;} v={.u=(uint32_t)acc<<16}; acc=bf(v.f+p); }
    }
    union {float f; uint32_t u;} v={.u=(uint32_t)acc<<16};
    out[j]=bf(ldexpf(v.f,scale));
  }
}
