#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#if !defined(__riscv) || __riscv_xlen != 64 || !defined(__riscv_float_abi_double)
#error RV64 lp64d required
#endif
static float (*volatile sine)(float)=sinf;
static float (*volatile cosine)(float)=cosf;
static uint32_t word(float x) {uint32_t b; memcpy(&b,&x,4); return b;}
int main(int argc,char **argv) {
 if(argc!=2)return 1;
 unsigned rm; __asm__ volatile("frrm %0":"=r"(rm)); if(rm)return 2;
 FILE *in=fopen(argv[1],"rb"); float freq[32];
 if(!in || fread(freq,4,32,in)!=32)return 3; fclose(in);
 FILE *out=stdout;
 for(int p=0;p<2048;p++)for(int j=0;j<32;j++) {
  volatile float angle=(float)p*freq[j];
  volatile float s=sine(angle),c=cosine(angle);
  volatile float ss=s*1.0f,cc=c*1.0f;
  uint32_t w[3]={word(angle),word(ss),word(cc)};
  if(fwrite(w,4,3,out)!=3)return 5;
 }
 return fclose(out)!=0;
}
